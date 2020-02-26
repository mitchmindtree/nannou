use nannou::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicBool};

struct Model {
    compute: Compute,
}

struct Compute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    buffer_len: usize,
    buffer_size: wgpu::BufferAddress,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    in_flight: Arc<AtomicBool>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(1440, 512)
        .title("nannou")
        .view(view)
        .build()
        .unwrap();

    let numbers = vec![1u32, 3, 5, 7];
    let size = (numbers.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    }).unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions::default(),
        limits: wgpu::Limits::default(),
    });

    let cs = include_bytes!("shaders/comp.spv");
    let cs_module = device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap());

    let staging_buffer = device
        .create_buffer_mapped(
            numbers.len(),
            wgpu::BufferUsage::MAP_READ
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        )
        .fill_from_slice(&numbers);

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer { dynamic: false, readonly: false },
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &storage_buffer,
                range: 0 .. size,
            },
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let in_flight = Arc::new(AtomicBool::new(false));

    let compute = Compute {
        device,
        queue,
        staging_buffer,
        storage_buffer,
        buffer_len: numbers.len(),
        buffer_size: size,
        bind_group,
        pipeline,
        in_flight,
    };

    Model { compute }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    println!("update");
    let compute = &mut model.compute;
    let device = &compute.device;

    // Only run the compute pass if there isn't already one in flight.
    if !compute.in_flight.load(atomic::Ordering::Relaxed) {
        // The encoder we'll use to encode the compute pass.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&compute.staging_buffer, 0, &compute.storage_buffer, 0, compute.buffer_size);
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute.pipeline);
            cpass.set_bind_group(0, &compute.bind_group, &[]);
            cpass.dispatch(compute.buffer_len as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&compute.storage_buffer, 0, &compute.staging_buffer, 0, compute.buffer_size);

        compute.queue.submit(&[encoder.finish()]);

        let in_flight_2 = compute.in_flight.clone();
        compute.in_flight.store(true, atomic::Ordering::Relaxed);
        compute.staging_buffer.map_read_async(0, compute.buffer_size, move |result: wgpu::BufferMapAsyncResult<&[u32]>| {
            if let Ok(mapping) = result {
                println!("Times: {:?}", mapping.data);
            }
            in_flight_2.store(false, atomic::Ordering::Relaxed);
        });
    }
    device.poll(false);
}

fn view(app: &App, model: &Model, frame: &Frame) {
    println!("view");
    frame.clear(CORNFLOWERBLUE);
}
