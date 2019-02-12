use audio::cpal;
use audio::sample::{Sample, ToSample};
use audio::stream;
use audio::{Buffer, Device, Requester, Stream};
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// The function that will be called when a `Buffer` is ready to be rendered.
pub trait RenderFn<M, S>: Fn(&mut M, &mut Buffer<S>) {}
impl<M, S, F> RenderFn<M, S> for F where F: Fn(&mut M, &mut Buffer<S>) {}

pub struct Builder<M = (), S = f32, F = fn(&mut M, &mut Buffer<S>)> {
    pub builder: super::Builder<M, S>,
    pub render: F,
}

/// An iterator yielding all available audio devices that support output streams.
pub struct Devices {
    pub(crate) devices: cpal::OutputDevices,
}

impl Iterator for Devices {
    type Item = Device;
    fn next(&mut self) -> Option<Self::Item> {
        self.devices.next().map(|device| Device { device })
    }
}

/// An empty function used as the default capture function if none was specified. 
pub fn default_render_fn<S>(_model: &mut (), _buffer: &mut Buffer<S>) {}

impl<M, S, F> Builder<M, S, F> {
    /// The "model" that represents the state of the program on the audio thread.
    pub fn model<M2>(self, model: M2) -> Builder<M2, S, F> {
        let Builder {
            render,
            builder: super::Builder {
                event_loop,
                process_fn_tx,
                sample_rate,
                channels,
                frames_per_buffer,
                device,
                sample_format,
                ..
            },
        } = self;
        Builder {
            render,
            builder: super::Builder {
                event_loop,
                process_fn_tx,
                model,
                sample_rate,
                channels,
                frames_per_buffer,
                device,
                sample_format,
            }
        }
    }

    /// Specify a function to use for handling buffers of audio input.
    pub fn render<F2, S2>(self, render: F2) -> Builder<M, S2, F2> {
        let Builder {
            builder: super::Builder {
                model,
                event_loop,
                process_fn_tx,
                sample_rate,
                channels,
                frames_per_buffer,
                device,
                ..
            },
            ..
        } = self;
        Builder {
            render,
            builder: super::Builder {
                model,
                event_loop,
                process_fn_tx,
                sample_rate,
                channels,
                frames_per_buffer,
                device,
                sample_format: PhantomData,
            }
        }
    }

    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        assert!(sample_rate > 0);
        self.builder.sample_rate = Some(sample_rate);
        self
    }

    pub fn channels(mut self, channels: usize) -> Self {
        assert!(channels > 0);
        self.builder.channels = Some(channels);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.builder.device = Some(device);
        self
    }

    pub fn frames_per_buffer(mut self, frames_per_buffer: usize) -> Self {
        assert!(frames_per_buffer > 0);
        self.builder.frames_per_buffer = Some(frames_per_buffer);
        self
    }

    pub fn build(self) -> Result<Stream<M>, super::BuildError>
    where
        S: 'static + Send + Sample + ToSample<u16> + ToSample<i16> + ToSample<f32>,
        M: 'static + Send,
        F: 'static + RenderFn<M, S> + Send,
    {
        let Builder {
            render,
            builder:
                stream::Builder {
                    event_loop,
                    process_fn_tx,
                    model,
                    sample_rate,
                    channels,
                    frames_per_buffer,
                    device,
                    ..
                },
        } = self;

        let sample_rate = sample_rate
            .map(|sr| cpal::SampleRate(sr))
            .or(Some(cpal::SampleRate(super::DEFAULT_SAMPLE_RATE)));
        let sample_format = super::cpal_sample_format::<S>();

        let device = match device {
            None => cpal::default_output_device().ok_or(super::BuildError::DefaultDevice)?,
            Some(Device { device }) => device,
        };

        // Find the best matching format.
        let format =
            super::find_best_matching_format(
                &device,
                sample_format,
                channels,
                sample_rate,
                device.default_output_format().ok(),
                |device| device.supported_output_formats().map(|fs| fs.collect()),
            )?.expect("no matching supported audio output formats for the target device");
        let stream_id = event_loop.build_output_stream(&device, &format)?;
        let (update_tx, update_rx) = mpsc::channel();
        let model = Arc::new(Mutex::new(Some(model)));
        let model_2 = model.clone();
        let num_channels = format.channels as usize;
        let sample_rate = format.sample_rate.0;

        // A buffer for collecting model updates.
        let mut pending_updates: Vec<Box<FnMut(&mut M) + 'static + Send>> = Vec::new();

        // Get the specified frames_per_buffer or fall back to a default.
        let frames_per_buffer = frames_per_buffer.unwrap_or(Buffer::<S>::DEFAULT_LEN_FRAMES);

        // An audio requester which requests frames from the model+render pair with a
        // specific buffer size, regardless of the buffer size requested by the OS.
        let mut requester = Requester::new(frames_per_buffer, num_channels);

        // An intermediary buffer for converting cpal samples to the target sample
        // format.
        let mut samples = vec![S::equilibrium(); frames_per_buffer * num_channels];

        // The function used to process a buffer of samples.
        let proc_output = move |data: cpal::StreamData| {
            // Collect and process any pending updates.
            macro_rules! process_pending_updates {
                () => {
                    // Collect any pending updates.
                    pending_updates.extend(update_rx.try_iter());

                    // If there are some updates available, take the lock and apply them.
                    if !pending_updates.is_empty() {
                        if let Ok(mut guard) = model_2.lock() {
                            let mut model = guard.take().unwrap();
                            for mut update in pending_updates.drain(..) {
                                update(&mut model);
                            }
                            *guard = Some(model);
                        }
                    }
                };
            }

            process_pending_updates!();

            // Retrieve the output buffer.
            let output = match data {
                cpal::StreamData::Output { mut buffer } => buffer,
                _ => unreachable!(),
            };

            samples.clear();
            samples.resize(output.len(), S::equilibrium());

            if let Ok(mut guard) = model_2.lock() {
                let mut m = guard.take().unwrap();
                m = requester.fill_buffer(m, &render, &mut samples, num_channels, sample_rate);
                *guard = Some(m);
            }

            // A function to simplify filling the unknown buffer type.
            fn fill_output<O, S>(output: &mut [O], buffer: &[S])
            where
                O: Sample,
                S: Sample + ToSample<O>,
            {
                for (out_sample, sample) in output.iter_mut().zip(buffer) {
                    *out_sample = sample.to_sample();
                }
            }

            // Process the given buffer.
            match output {
                cpal::UnknownTypeOutputBuffer::U16(mut buffer) => {
                    fill_output(&mut buffer, &samples);
                }
                cpal::UnknownTypeOutputBuffer::I16(mut buffer) => {
                    fill_output(&mut buffer, &samples);
                }
                cpal::UnknownTypeOutputBuffer::F32(mut buffer) => {
                    fill_output(&mut buffer, &samples)
                }
            }

            process_pending_updates!();
        };

        // Send the buffer processing function to the event loop.
        process_fn_tx
            .send((stream_id.clone(), Box::new(proc_output)))
            .unwrap();

        let shared = Arc::new(super::Shared {
            model,
            stream_id,
            event_loop,
            is_paused: AtomicBool::new(false),
        });

        let stream = Stream {
            shared,
            process_fn_tx,
            update_tx,
            cpal_format: format,
        };
        Ok(stream)
    }
}
