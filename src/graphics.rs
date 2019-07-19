use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use specs::prelude::*;

use image::{DynamicImage, ImageBuffer, Rgba};

pub use image::ImageFormat;

use rusttype::{point, Font, Scale};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{DescriptorSet, FixedSizeDescriptorSetsPool};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{Dimensions, ImmutableImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder};

use super::core::{GameObjectComponent, MethodAdder, Scriptable};
use super::position::Position;

lazy_static! {
    static ref VIEWPORT_SIZE: Mutex<(u32, u32)> = Mutex::new((0, 0));
}

fn load_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path)?))
}

#[derive(Clone)]
pub struct Graphics {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    position: Option<(f64, f64)>,
    scale: f64,
    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
    texture_buffer: Option<Arc<DescriptorSet + Send + Sync>>,
}

impl Graphics {
    pub fn from_image(image: DynamicImage) -> Graphics {
        Graphics {
            image: image.to_rgba(),
            position: None,
            scale: 1.0,
            vertex_buffer: None,
            texture_buffer: None,
        }
    }

    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        format: ImageFormat,
    ) -> std::io::Result<Graphics> {
        Graphics::load_with_scale(path, format, 1.0)
    }

    pub fn load_with_crop<P: AsRef<std::path::Path>>(
        path: P,
        format: ImageFormat,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> std::io::Result<Graphics> {
        Graphics::load_with_scale_and_crop(path, format, 1.0, x, y, width, height)
    }

    pub fn load_with_scale<P: AsRef<std::path::Path>>(
        path: P,
        format: ImageFormat,
        scale: f64,
    ) -> std::io::Result<Graphics> {
        let image = image::load(load_file(path)?, format).unwrap().to_rgba();

        Ok(Graphics {
            image: image,
            position: None,
            scale: scale,
            vertex_buffer: None,
            texture_buffer: None,
        })
    }

    pub fn load_with_scale_and_crop<P: AsRef<std::path::Path>>(
        path: P,
        format: ImageFormat,
        scale: f64,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> std::io::Result<Graphics> {
        let image = image::load(load_file(path)?, format)
            .unwrap()
            .crop(x, y, width, height)
            .to_rgba();

        Ok(Graphics {
            image: image,
            position: None,
            scale: scale,
            vertex_buffer: None,
            texture_buffer: None,
        })
    }

    pub fn load_text_with_font_path<P: AsRef<std::path::Path>>(
        text: String,
        font_path: P,
        color: (u8, u8, u8),
        size: f32,
    ) -> std::io::Result<Graphics> {
        let mut font_data = Vec::new();
        load_file(font_path)?.read_to_end(&mut font_data)?;
        Ok(Graphics::load_text(
            text,
            &Font::from_bytes(&font_data).unwrap(),
            color,
            size,
        ))
    }

    pub fn load_text(text: String, font: &Font, color: (u8, u8, u8), size: f32) -> Graphics {
        let scale = Scale::uniform(size);
        let v_metrics = font.v_metrics(scale);

        let glyphs: Vec<_> = font
            .layout(&text, scale, point(20.0, 20.0 + v_metrics.ascent))
            .collect();

        let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;

        let glyphs_width = {
            let min_x = glyphs
                .first()
                .map(|g| g.pixel_bounding_box().unwrap().min.x)
                .unwrap();
            let max_x = glyphs
                .last()
                .map(|g| g.pixel_bounding_box().unwrap().max.x)
                .unwrap();
            (max_x - min_x) as u32
        };

        let mut image = DynamicImage::new_rgba8(glyphs_width + 40, glyphs_height + 40).to_rgba();

        for glyph in glyphs {
            if let Some(bounding_box) = glyph.pixel_bounding_box() {
                // Draw the glyph into the image per-pixel by using the draw closure
                glyph.draw(|x, y, v| {
                    image.put_pixel(
                        // Offset the position by the glyph bounding box
                        x + bounding_box.min.x as u32,
                        y + bounding_box.min.y as u32,
                        // Turn the coverage into an alpha value
                        Rgba {
                            data: [color.0, color.1, color.2, (v * 255.0) as u8],
                        },
                    )
                });
            }
        }

        Graphics {
            image: image,
            position: None,
            scale: 1.0,
            vertex_buffer: None,
            texture_buffer: None,
        }
    }

    fn create_vertexes_for_position(
        &self,
        (width, height): (f64, f64),
        (window_width, window_height): (f64, f64),
        (x, y): (f64, f64),
    ) -> (f32, f32, f32, f32) {
        let lower_x = ((x - width / 2.0) / window_width * self.scale) as f32;
        let upper_x = ((x + width / 2.0) / window_width * self.scale) as f32;
        let lower_y = ((y - height / 2.0) / window_height * self.scale) as f32;
        let upper_y = ((y + height / 2.0) / window_height * self.scale) as f32;
        (lower_x, upper_x, lower_y, upper_y)
    }

    fn create_vertex_buffer(
        &self,
        position: (f64, f64),
        device: Arc<Device>,
    ) -> Result<Arc<CpuAccessibleBuffer<[Vertex]>>, Box<dyn Error>> {
        let reg_dimensions = self.image.dimensions();
        let dimensions = (reg_dimensions.0 as f64, reg_dimensions.1 as f64);
        let viewport = VIEWPORT_SIZE.lock().unwrap().clone();
        let window_dimensions = (viewport.0 as f64, viewport.1 as f64);
        let (lower_x, upper_x, lower_y, upper_y) =
            self.create_vertexes_for_position(dimensions, window_dimensions, position);

        let buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            device,
            BufferUsage::all(),
            [
                Vertex {
                    position: [lower_x, lower_y, 0.0, 0.0],
                },
                Vertex {
                    position: [upper_x, lower_y, 1.0, 0.0],
                },
                Vertex {
                    position: [lower_x, upper_y, 0.0, 1.0],
                },
                Vertex {
                    position: [upper_x, upper_y, 1.0, 1.0],
                },
            ]
            .iter()
            .cloned(),
        )?;
        Ok(buffer)
    }

    fn load_texture_into_gpu(
        &self,
        graphics: &GraphicsSystem,
    ) -> Result<
        (
            Arc<DescriptorSet + Send + Sync>,
            Box<GpuFuture + Send + Sync>,
        ),
        Box<Error>,
    > {
        let dimensions = self.image.dimensions();
        let image_dimensions = (dimensions.0 as f64, dimensions.1 as f64);

        let image_data = self.image.clone().into_raw();
        let (texture, tex_future) = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            Dimensions::Dim2d {
                width: image_dimensions.0 as u32,
                height: image_dimensions.1 as u32,
            },
            Format::R8G8B8A8Srgb,
            graphics.queue.clone(),
        )?;

        let set = Arc::new(
            graphics
                .descriptor_pool
                .lock()
                .unwrap()
                .next()
                .add_sampled_image(texture, graphics.sampler.clone())?
                .build()?,
        );

        Ok((set, Box::new(tex_future)))
    }

    fn set_position(
        &mut self,
        (new_x, new_y): (f64, f64),
        device: Arc<Device>,
    ) -> Result<(), Box<Error>> {
        let (width, height) = *VIEWPORT_SIZE.lock().unwrap();
        match self.position {
            Some((old_x, old_y)) => {
                self.position = Some((new_x, new_y));
                let (delta_x, delta_y) = (
                    (new_x - old_x) / width as f64,
                    (new_y - old_y) / height as f64,
                );
                if let Some(vertex_buffer) = &self.vertex_buffer {
                    match vertex_buffer.write() {
                        Ok(mut vertexes) => {
                            for vertex in vertexes.iter_mut() {
                                let (vertex_x, vertex_y) = (vertex.position[0], vertex.position[1]);
                                vertex.position[0] = vertex_x + delta_x as f32;
                                vertex.position[1] = vertex_y + delta_y as f32;
                            }
                        }
                        Err(e) => error!("Failed to lock CPU buffer: {}", e),
                    }
                }
            }
            None => {
                self.position = Some((new_x, new_y));
                self.vertex_buffer = Some(self.create_vertex_buffer((new_x, new_y), device)?);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Vertex {
    position: [f32; 4],
}
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
    pipeline: Arc<
        GraphicsPipeline<
            SingleBufferDefinition<Vertex>,
            Box<dyn PipelineLayoutAbstract + Send + Sync>,
            Arc<dyn RenderPassAbstract + Send + Sync>,
        >,
    >,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    descriptor_pool: Arc<
        Mutex<FixedSizeDescriptorSetsPool<Arc<PipelineLayoutAbstract + Send + Sync + 'static>>>,
    >,
    sampler: Arc<Sampler>,
}

#[derive(Debug)]
struct NoneError;

impl std::fmt::Display for NoneError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "No value found")
    }
}

#[derive(Debug)]
struct NoWindowError;

impl std::error::Error for NoneError {}

impl std::fmt::Display for NoWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "No open window found")
    }
}

impl std::error::Error for NoWindowError {}

impl GraphicsSystem {
    pub fn new(
        events_loop: &EventsLoop,
        swaphchain_flag: Arc<AtomicBool>,
    ) -> Result<GraphicsSystem, Box<std::error::Error>> {
        // The start of this example is exactly the same as `triangle`. You should read the
        // `triangle` example if you haven't done so yet.

        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &extensions, None)?;

        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .ok_or(NoneError)?;
        trace!(
            "Using device: {} (type: {:?})",
            physical.name(),
            physical.ty()
        );

        let surface = WindowBuilder::new().build_vk_surface(events_loop, instance.clone())?;
        let window = surface.window();

        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )?;
        let queue = queues.next().ok_or(NoneError)?;

        *VIEWPORT_SIZE.lock().unwrap() = surface.window().get_inner_size().unwrap().into();

        let (swapchain, images) = {
            let caps = surface.capabilities(physical)?;

            let usage = caps.supported_usage_flags;
            let alpha = caps
                .supported_composite_alpha
                .iter()
                .next()
                .ok_or(NoneError)?;
            let format = caps.supported_formats[0].0;

            let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
                // convert to physical pixels
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                // The window no longer exists so exit the application.
                return Err(Box::new(NoWindowError));
            };

            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                initial_dimensions,
                1,
                usage,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                true,
                None,
            )?
        };

        let vs = vs::Shader::load(device.clone())?;
        let fs = fs::Shader::load(device.clone())?;

        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
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
        )?);

        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0,
        )?;

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(
                    Subpass::from(
                        render_pass.clone() as Arc<RenderPassAbstract + Send + Sync>,
                        0,
                    )
                    .ok_or(NoneError)?,
                )
                .build(device.clone())?,
        );

        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
        };
        let framebuffers =
            window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

        let descriptor_pool = Arc::new(Mutex::new(FixedSizeDescriptorSetsPool::new(
            pipeline.clone() as Arc<PipelineLayoutAbstract + Send + Sync>,
            0,
        )));

        Ok(GraphicsSystem {
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
            sampler: sampler,
        })
    }

    fn build_render_pass<'a>(
        &mut self,
        mut graphics: WriteStorage<'a, GameObjectComponent<Graphics>>,
        positions: ReadStorage<'a, GameObjectComponent<Position>>,
        image_num: usize,
    ) -> Result<(AutoCommandBuffer, Box<GpuFuture + Send + Sync>), Box<std::error::Error>> {
        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
        let mut cb_in_progress = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )?
        .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values)?;

        let mut previous_frame_end = std::mem::replace(
            &mut self.previous_frame_end,
            Box::new(sync::now(self.device.clone())),
        );

        for (graphics_data, position_data) in (&mut graphics, &positions).join() {
            let mut data = graphics_data.get();
            let new_position = position_data.get().get();
            if data.position != Some(new_position) {
                data.set_position(new_position, self.device.clone())?;
            }
            let position = data.position.unwrap();
            let vertex_buffer = data.vertex_buffer.clone().unwrap_or_else(|| {
                let vertex_buffer = data
                    .create_vertex_buffer(position, self.device.clone())
                    .expect("Failed to create vertex buffer");
                data.vertex_buffer = Some(vertex_buffer.clone());
                vertex_buffer
            });

            let descriptor_set = match data.texture_buffer.clone() {
                Some(s) => s,
                None => {
                    let (texture_buffer, tex_future) = data.load_texture_into_gpu(self)?;
                    previous_frame_end = Box::new(previous_frame_end.join(tex_future));
                    data.texture_buffer = Some(texture_buffer.clone());
                    texture_buffer
                }
            };

            cb_in_progress = cb_in_progress.draw(
                self.pipeline.clone(),
                &self.dynamic_state,
                vertex_buffer,
                descriptor_set,
                (),
            )?;
        }
        Ok((
            cb_in_progress.end_render_pass()?.build()?,
            previous_frame_end,
        ))
    }
}

impl Scriptable for Graphics {
    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(_methods: &mut M) {}

    fn name() -> &'static str {
        "image"
    }
}

impl<'a> System<'a> for GraphicsSystem {
    type SystemData = (
        ReadStorage<'a, GameObjectComponent<Position>>,
        WriteStorage<'a, GameObjectComponent<Graphics>>,
    );

    fn run(&mut self, (graphics, positions): Self::SystemData) {
        let window = self.surface.window();
        self.previous_frame_end.cleanup_finished();
        if self.recreate_swapchain.load(Ordering::Relaxed) {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                *VIEWPORT_SIZE.lock().unwrap() = dimensions;
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) =
                match self.swapchain.recreate_with_dimension(dimensions) {
                    Ok(r) => r,
                    Err(err) => {
                        error!("Failed to recreate swapchain: {}", err);
                        return;
                    }
                };

            self.swapchain = new_swapchain;
            self.framebuffers = window_size_dependent_setup(
                &new_images,
                self.render_pass.clone(),
                &mut self.dynamic_state,
            );

            self.recreate_swapchain.store(false, Ordering::Relaxed);
        }

        let (image_num, future) = match swapchain::acquire_next_image(self.swapchain.clone(), None)
        {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain.store(true, Ordering::Relaxed);
                return;
            }
            Err(err) => {
                error!("Failed to aquire next swapchain image: {}", err);
                return;
            }
        };

        let (cb, previous_frame_end) = match self.build_render_pass(positions, graphics, image_num)
        {
            Ok(cb) => cb,
            Err(e) => {
                error!("Failed to build render pass: {}", e);
                return;
            }
        };

        let future = previous_frame_end
            .join(future)
            .then_execute(self.queue.clone(), cb)
            .unwrap()
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

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
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
    vulkano_shaders::shader! {
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
