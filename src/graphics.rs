// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use specs::prelude::*;

use image::ImageBuffer;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::descriptor::descriptor_set::{DescriptorSet, FixedSizeDescriptorSetsPool};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, ImmutableImage, Dimensions, ImageCreationError};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::sampler::{BorderColor, Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, Surface, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder};

use image::ImageFormat;

use std::sync::Arc;

fn load_image<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path)?))
}

pub struct Graphics {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    position: (f64, f64),
    scale: f64,
    data: Option<(Arc<CpuAccessibleBuffer<[Vertex]>>, Arc<DescriptorSet + Send + Sync>)>
}

impl Graphics {
    pub fn load<P: AsRef<std::path::Path>>(path: P, format: ImageFormat, x: f64, y: f64) -> std::io::Result<Graphics> {
        Graphics::load_with_scale(path, format, x, y, 1.0)
    }

    pub fn load_with_scale<P: AsRef<std::path::Path>>(path: P, format: ImageFormat, x: f64, y: f64, scale: f64) -> std::io::Result<Graphics> {
        let image = image::load(load_image(path)?, format).unwrap().to_rgba();

        Ok(Graphics{
            image: image,
            position: (x, y),
            scale: scale,
            data: None
        })
    }

    fn do_load(&mut self, graphics: &mut GraphicsSystem) -> (Box<GpuFuture + Send + Sync>, Arc<CpuAccessibleBuffer<[Vertex]>>, Arc<DescriptorSet + Send + Sync>) {
        let dimensions = self.image.dimensions();
        let (window_width, window_height): (f64, f64) = graphics.surface.window().get_inner_size().unwrap().into();
        let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
        let (x, y) = (self.position.0 as f64, self.position.1 as f64);

        let lower_x = ((x - width/2.0)/window_width * self.scale) as f32;
        let upper_x = ((x + width/2.0)/window_width * self.scale) as f32;
        let lower_y = ((y - height/2.0)/window_height * self.scale) as f32;
        let upper_y = ((y + height/2.0)/window_height * self.scale) as f32;

        let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            graphics.device.clone(),
            BufferUsage::all(),
            [
                Vertex { position: [ lower_x, lower_y, 0.0, 0.0] },
                Vertex { position: [ upper_x, lower_y,  1.0, 0.0] },
                Vertex { position: [ lower_x, upper_y, 0.0,  1.0] },
                Vertex { position: [ upper_x, upper_y,  1.0,  1.0] },
            ].iter().cloned()
        ).unwrap();

        let image_data = self.image.clone().into_raw();

        let (texture, tex_future) = ImmutableImage::from_iter(
            image_data.iter().cloned(),
                Dimensions::Dim2d { width: width as u32, height: height as u32 },
                Format::R8G8B8A8Srgb,
                graphics.queue.clone()
        ).unwrap();

        let set = Arc::new(graphics.descriptor_pool.lock().unwrap().next()
            .add_sampled_image(texture, graphics.sampler.clone()).unwrap()
            .build().unwrap());

        self.data = Some((vertex_buffer.clone(), set.clone()));

        (Box::new(tex_future), vertex_buffer, set)
    }
}

#[derive(Debug, Clone)]
struct Vertex { position: [f32; 4]}
vulkano::impl_vertex!(Vertex, position);

impl Component for Graphics {
    type Storage = VecStorage<Self>;
}

pub struct GraphicsSystem {
    recreate_swapchain: Arc<AtomicBool>,
    previous_frame_end: Box<GpuFuture + Send + Sync>,
    surface: Arc<Surface<Window>>,
    swapchain: Arc<Swapchain<Window>>,
    queue: Arc<Queue>,
    device: Arc<Device>,
    dynamic_state: DynamicState,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    pipeline: Arc<GraphicsPipeline<
        SingleBufferDefinition<Vertex>,
        Box<dyn PipelineLayoutAbstract + Send + Sync>,
        Arc<dyn RenderPassAbstract + Send + Sync>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    descriptor_pool: Arc<Mutex<FixedSizeDescriptorSetsPool<Arc<PipelineLayoutAbstract + Send + Sync + 'static>>>>,
    sampler: Arc<Sampler>
}

impl GraphicsSystem {

    pub fn new(events_loop: &EventsLoop, swaphchain_flag: Arc<AtomicBool>) -> Result<GraphicsSystem, ()> {
        // The start of this example is exactly the same as `triangle`. You should read the
        // `triangle` example if you haven't done so yet.

        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &extensions, None).unwrap();

        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        trace!("Using device: {} (type: {:?})", physical.name(), physical.ty());

        let surface = WindowBuilder::new().build_vk_surface(events_loop, instance.clone()).unwrap();
        let window = surface.window();

        let queue_family = physical.queue_families().find(|&q|
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        ).unwrap();

        let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
        let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
            [(queue_family, 0.5)].iter().cloned()).unwrap();
        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();

            let usage = caps.supported_usage_flags;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;

            let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
                // convert to physical pixels
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                // The window no longer exists so exit the application.
                return Err(());
            };

            Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
                PresentMode::Fifo, true, None).unwrap()
        };

        let vs = vs::Shader::load(device.clone()).unwrap();
        let fs = fs::Shader::load(device.clone()).unwrap();

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            ).unwrap()
        );

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0, 1.0, 0.0, 0.0).unwrap();

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .blend_alpha_blending()
            .render_pass(Subpass::from(render_pass.clone() as Arc<RenderPassAbstract + Send + Sync>, 0).unwrap())
            .build(device.clone())
            .unwrap());

        let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };
        let framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

        let descriptor_pool = Arc::new(Mutex::new(FixedSizeDescriptorSetsPool::new(pipeline.clone() as Arc<PipelineLayoutAbstract + Send + Sync>, 0)));

        Ok(GraphicsSystem{
            recreate_swapchain: swaphchain_flag,
            device: device.clone(),
            previous_frame_end: Box::new(sync::now(device.clone())),
            surface: surface,
            swapchain: swapchain,
            dynamic_state: dynamic_state,
            framebuffers: framebuffers,
            pipeline: pipeline,
            queue: queue,
            render_pass: render_pass,
            descriptor_pool: descriptor_pool,
            sampler: sampler
        })
    }
}

impl<'a> System<'a> for GraphicsSystem {
    type SystemData = WriteStorage<'a, Graphics>;

    fn run(&mut self, mut data: Self::SystemData) {
        let window = self.surface.window();
        self.previous_frame_end.cleanup_finished();
        if self.recreate_swapchain.load(Ordering::Relaxed) {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err)
            };

            self.swapchain = new_swapchain;
            self.framebuffers = window_size_dependent_setup(&new_images, self.render_pass.clone(), &mut self.dynamic_state);

            self.recreate_swapchain.store(false, Ordering::Relaxed);
        }

        let (image_num, future) = match swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain.store(true, Ordering::Relaxed);
                return;
            }
            Err(err) => panic!("{:?}", err)
        };

        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());
        let mut cb_in_progress = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
            .unwrap()
            .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values).unwrap();

        let mut previous_frame_end = std::mem::replace(&mut self.previous_frame_end, Box::new(sync::now(self.device.clone())));

        for graphics_data in (&mut data).join() {
            let (vertex_buffer, descriptor_set) = match graphics_data.data.clone() {
                Some(data) => data,
                None => {
                    let (tex_future, vertex_buffer, set) = graphics_data.do_load(self);
                    previous_frame_end = Box::new(previous_frame_end.join(tex_future));

                    (vertex_buffer, set)
                }
            };

            cb_in_progress = cb_in_progress.draw(self.pipeline.clone(),
                &self.dynamic_state,
                vertex_buffer,
                descriptor_set,
                ()).unwrap();
                // previous_frame_end = Box::new(previous_frame_end.join(std::mem::replace(
                //             &mut graphics_data.future,
                //             Box::new(sync::now(self.device.clone())))));
        }
        let cb = cb_in_progress.end_render_pass().unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(future)
            .then_execute(self.queue.clone(), cb).unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                if cfg!(target_os = "macos") {
                    // Workaround for moltenvk issue (hang on close)
                    // FIXME Remove once motenvk is fixed
                    future.wait(None).expect("waiting on fence failed");
                }
                self.previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                  self.recreate_swapchain.store(true, Ordering::Relaxed);
                self.previous_frame_end = Box::new(sync::now(self.device.clone())) as Box<_>;
            }
            Err(e) => {
                error!("{:?}", e);
                self.previous_frame_end = Box::new(sync::now(self.device.clone())) as Box<_>;
            }
        }
    }
}

pub fn render() {
    // loop {

    //     let mut done = false;
    //     events_loop.poll_events(|ev| {
    //         match ev {
    //             Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
    //             Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
    //             _ => ()
    //         }
    //     });
    //     if done { return; }
    // }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec4 position;
layout(location = 0) out vec2 tex_coords;
void main() {
    gl_Position = vec4(vec2(position), 0.0, 1.0);
    tex_coords = vec2(position[2], position[3]);
}"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) uniform sampler2D tex;
void main() {
    f_color = texture(tex, tex_coords);
}"
    }
}