use nannou::prelude::*;
use nannou_egui::Egui;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    egui: Egui,
    egui_demo_windows: egui_demo_lib::DemoWindows,
}

fn model(app: &App) -> Model {
    app.set_loop_mode(LoopMode::wait());
    let w_id = app
        .new_window()
        .raw_event(raw_window_event)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(w_id).unwrap();
    let egui = Egui::from_window(&window);
    Model {
        egui,
        egui_demo_windows: Default::default(),
    }
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn update(_app: &App, model: &mut Model, update: Update) {
    let Model {
        ref mut egui,
        ref mut egui_demo_windows,
        ..
    } = *model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    egui_demo_windows.ui(&ctx);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    draw.to_frame(app, &frame).unwrap();

    model.egui.draw_to_frame(&frame);
}
