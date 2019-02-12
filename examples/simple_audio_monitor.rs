//! Setup an input stream and monitor the audio signal.
//!
//! Note that while this example demonstrates monitoring audio with an input stream, you can also
//! monitor the output stream using the same approach.

extern crate nannou;

use nannou::audio::{self, Buffer};
use nannou::prelude::*;

fn main() {
    nannou::app(model).simple_window(view).run();
}

struct Model {
    stream: audio::Stream<()>,
}

fn model(app: &App) -> Model {
    let stream = app.new_input_stream().build().unwrap();
    Model { stream }
}

fn view(_app: &App, model: &Model, frame: Frame) -> Frame {
    let rms = model.stream.peak_rms();
    let [low, mid, high] = model.stream.peak_fft_3_band();

    frame.clear(BLACK);
    frame
}
