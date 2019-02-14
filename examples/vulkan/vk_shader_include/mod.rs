extern crate nannou;

use nannou::prelude::*;
use nannou::vulkano;
use std::cell::RefCell;
use std::sync::Arc;

use nannou::vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use nannou::vulkano::command_buffer::DynamicState;
use nannou::vulkano::device::DeviceOwned;
use nannou::vulkano::framebuffer::{RenderPassAbstract, Subpass};
use nannou::vulkano::pipeline::viewport::Viewport;
use nannou::vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};

fn main() {
    nannou::app(model).run();
}

struct Model {
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    framebuffer: RefCell<ViewFramebuffer>,
}

#[derive(Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}

nannou::vulkano::impl_vertex!(Vertex, position);

fn model(app: &App) -> Model {
    app.new_window()
        .with_dimensions(1024, 512)
        .with_title("nannou")
        .view(view)
        .build()
        .unwrap();

    let device = app.main_window().swapchain().device().clone();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        [
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    let vertex_shader = vs::Shader::load(device.clone()).unwrap();
    let fragment_shader = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        nannou::vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: app.main_window().swapchain().format(),
                    samples: app.main_window().msaa_samples(),
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .blend_alpha_blending()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let framebuffer = RefCell::new(ViewFramebuffer::default());

    Model {
        render_pass,
        pipeline,
        vertex_buffer,
        framebuffer,
    }
}

fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    let [w, h] = frame.swapchain_image().dimensions();
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [w as _, h as _],
        depth_range: 0.0..1.0,
    };
    let dynamic_state = DynamicState {
        line_width: None,
        viewports: Some(vec![viewport]),
        scissors: None,
    };

    // Update the framebuffer in case of window resize.
    model.framebuffer.borrow_mut()
        .update(&frame, model.render_pass.clone(), |builder, image| builder.add(image))
        .unwrap();

    let clear_values = vec![[0.0, 1.0, 0.0, 1.0].into()];

    let push_constants = fs::ty::PushConstantData {
        time: app.time,
        width: w as f32,
        height: h as f32,
    };

    frame
        .add_commands()
        .begin_render_pass(
            model.framebuffer.borrow().as_ref().unwrap().clone(),
            false,
            clear_values,
        )
        .unwrap()
        .draw(
            model.pipeline.clone(),
            &dynamic_state,
            vec![model.vertex_buffer.clone()],
            (),
            push_constants,
        )
        .unwrap()
        .end_render_pass()
        .expect("failed to add `end_render_pass` command");

    frame
}

mod vs {
    nannou::vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}"
    }
}

mod fs {
    nannou::vulkano_shaders::shader! {
    ty: "fragment",
    // We declare what directories to search for when using the `#include <...>`
    // syntax. Specified directories have descending priorities based on their order.
    include: [ "examples/vulkan/vk_shader_include/common_shaders" ],
        src: "
#version 450
// Substitutes this line with the contents of the file `lfos.glsl` found in one of the standard
// `include` directories specified above.
// Note, that relative inclusion (`#include \"...\"`), although it falls back to standard
// inclusion, should not be used for **embedded** shader source, as it may be misleading and/or
// confusing.
#include <lfos.glsl>

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PushConstantData {
    float time;
    float width;
    float height;
} pc;

float circle(in vec2 _st, in float _radius){
    vec2 dist = _st-vec2(0.5);
	return 1.-smoothstep(_radius-(_radius*0.01),
                         _radius+(_radius*0.01),
                         dot(dist,dist)*4.0);
}

void main() {

    // Background
	vec3 bg = vec3(0.8,0.9,0.9);

    float aspect = pc.width / pc.height;
    vec2 center = vec2(0.5,0.5);
    float radius = 0.25 * aspect;

    // Here we use the 'lfo' function imported from our include shader
    float x = tex_coords.x + lfo(2, pc.time * 3.0) * 0.7;
    vec3 color = vec3(lfo(3,pc.time * 6.0),0.0,0.9);
    vec3 shape = vec3(circle(vec2((x * aspect) - 0.45, tex_coords.y), radius) );
    shape *= color;

    // Blend the two
	f_color = vec4( vec3(mix(shape, bg, 0.5)),1.0 );
}"
    }
}
