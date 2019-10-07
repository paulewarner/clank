use std::convert::TryInto;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use specs::prelude::*;

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};

pub use image::ImageFormat;

use rusttype::{point, Font, Scale};

use nalgebra as na;

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

fn remap(x: f32, from_lo: f32, from_hi: f32, to_lo: f32, to_hi: f32) -> f32 {
    to_lo + (x - from_lo) * (to_hi - to_lo) / (from_hi - from_lo)
}

fn viewport_matrix(width: f32, height: f32) -> na::Matrix3<f32> {
    na::Matrix3::new(
        width / 2.0,
        0.0,
        0.0,
        0.0,
        height / 2.0,
        0.0,
        (width - 1.0) / 2.0,
        (height - 1.0) / 2.0,
        1.0,
    )
}

fn rotation_matrix(angle: f32) -> na::Matrix3<f32> {
    na::Matrix3::new(
        angle.to_radians().cos(),
        angle.to_radians().sin(),
        0.0,
        -angle.to_radians().sin(),
        angle.to_radians().cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

fn flip_matrix(flipped_horizontally: bool, flipped_vertically: bool) -> na::Matrix3<f32> {
    let x = match flipped_horizontally {
        true => -1.0,
        false => 1.0,
    };
    let y = match flipped_vertically {
        true => -1.0,
        false => 1.0,
    };
    na::Matrix3::new(x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 1.0)
}

fn translation_matrix(x: f32, y: f32) -> na::Matrix3<f32> {
    na::Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0)
}

fn scale_matrix(x_scale: f32, y_scale: f32) -> na::Matrix3<f32> {
    na::Matrix3::new(x_scale, 0.0, 0.0, 0.0, y_scale, 0.0, 0.0, 0.0, 1.0)
}

#[derive(Clone)]
pub struct Graphics {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    position: Option<(f32, f32)>,
    scale: f32,
    rotation: f32,
    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
    texture_buffer: Option<Arc<dyn DescriptorSet + Send + Sync>>,
    texture_position: (f32, f32),
    texture_size: (f32, f32),
    flipped_horizontally: bool,
    flipped_vertically: bool,
}

pub struct GraphicsBuilder {
    position: Option<(f32, f32)>,
    size: Option<(f32, f32)>,
    scale: Option<f32>,
    rotation: Option<f32>,
    texture_position: Option<(f32, f32)>,
    texture_size: Option<(f32, f32)>,
    image: Option<ImageBuffer<image::Rgba<u8>, Vec<u8>>>,
    native_size: Option<(f32, f32)>,
    flipped_horizontally: bool,
    flipped_vertically: bool,
}

impl GraphicsBuilder {
    pub fn position(mut self, x: f32, y: f32) -> Self {
        self.position = Some((x, y));
        self
    }

    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.size = Some((width, height));
        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }

    pub fn rotation(mut self, rotation: f32) -> Self {
        self.rotation = Some(rotation);
        self
    }

    pub fn texture_position(mut self, x: f32, y: f32) -> Self {
        self.texture_position = Some((x, y));
        self
    }

    pub fn texture_size(mut self, width: f32, height: f32) -> Self {
        self.texture_size = Some((width, height));
        self
    }

    pub fn flipped_horizontally(mut self) -> Self {
        self.flipped_horizontally = true;
        self
    }

    pub fn flipped_vertically(mut self) -> Self {
        self.flipped_vertically = true;
        self
    }

    pub fn image(mut self, image: DynamicImage) -> Self {
        let (width, height) = image.dimensions();
        self.native_size = Some((width as f32, height as f32));
        self.image = Some(image.to_rgba());
        self
    }

    pub fn load_image<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
        format: ImageFormat,
    ) -> std::io::Result<Self> {
        let image = image::load(load_file(path)?, format).unwrap().to_rgba();
        let (width, height) = image.dimensions();
        self.native_size = Some((width as f32, height as f32));
        self.image = Some(image);
        Ok(self)
    }

    pub fn text_with_font<S: AsRef<str>, P: AsRef<std::path::Path>>(
        self,
        text: S,
        font: P,
        color: (u8, u8, u8),
        size: f32,
    ) -> std::io::Result<Self> {
        let mut font_data = Vec::new();
        load_file(font)?.read_to_end(&mut font_data)?;
        Ok(self.text(text, &Font::from_bytes(&font_data).unwrap(), color, size))
    }

    pub fn text<S: AsRef<str>>(
        mut self,
        text: S,
        font: &Font,
        color: (u8, u8, u8),
        size: f32,
    ) -> Self {
        let scale = Scale::uniform(size);
        let v_metrics = font.v_metrics(scale);
        let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;

        let glyphs: Vec<_> = text
            .as_ref()
            .split('\n')
            .enumerate()
            .flat_map(|(n, line)| {
                font.layout(
                    line,
                    scale,
                    point(20.0, 20.0 + v_metrics.ascent + advance_height * n as f32),
                )
            })
            .collect();

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

        let glyphs_height = {
            let min_y = glyphs
                .first()
                .map(|g| g.pixel_bounding_box().unwrap().min.y)
                .unwrap();
            let max_y = glyphs
                .last()
                .map(|g| g.pixel_bounding_box().unwrap().max.y)
                .unwrap();
            (max_y - min_y) as u32
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
                        Rgba([color.0, color.1, color.2, (v * 255.0) as u8]),
                    )
                });
            }
        }

        let (width, height) = image.dimensions();

        self.image = Some(image);
        self.native_size = Some((width as f32, height as f32));
        self
    }

    pub fn build(self) -> Graphics {
        Graphics {
            image: self.image.unwrap(),
            position: self.position,
            scale: self.scale.unwrap_or(1.0),
            texture_position: self.texture_position.unwrap_or((0.0, 0.0)),
            texture_size: self.texture_size.or(self.native_size).unwrap(),
            texture_buffer: None,
            vertex_buffer: None,
            rotation: self.rotation.unwrap_or(0.0),
            flipped_horizontally: self.flipped_horizontally,
            flipped_vertically: self.flipped_vertically,
        }
    }
}

impl Graphics {
    pub fn new() -> GraphicsBuilder {
        GraphicsBuilder {
            position: None,
            size: None,
            scale: None,
            rotation: None,
            texture_position: None,
            texture_size: None,
            image: None,
            native_size: None,
            flipped_horizontally: false,
            flipped_vertically: false,
        }
    }

    fn create_vertexes_for_position(
        &self,
        (width, height): (f32, f32),
        (window_width, window_height): (f32, f32),
        (x, y): (f32, f32),
        scale: f32,
        lower_bound: f32,
        upper_bound: f32,
    ) -> (f32, f32, f32, f32) {
        let lower_x = remap(
            x - width / 2.0 * scale,
            -window_width / 2.0,
            window_width / 2.0,
            lower_bound,
            upper_bound,
        );
        let upper_x = remap(
            x + width / 2.0 * scale,
            -window_width / 2.0,
            window_width / 2.0,
            lower_bound,
            upper_bound,
        );
        let lower_y = remap(
            y - height / 2.0 * scale,
            -window_height / 2.0,
            window_height / 2.0,
            lower_bound,
            upper_bound,
        );
        let upper_y = remap(
            y + height / 2.0 * scale,
            -window_height / 2.0,
            window_height / 2.0,
            lower_bound,
            upper_bound,
        );
        (lower_x, upper_x, lower_y, upper_y)
    }

    fn create_vertex_buffer(
        &self,
        position: (f32, f32),
        device: Arc<Device>,
    ) -> Result<Arc<CpuAccessibleBuffer<[Vertex]>>, Box<dyn Error>> {
        let reg_dimensions = self.image.dimensions();
        let dimensions = (reg_dimensions.0 as f32, reg_dimensions.1 as f32);
        let (viewport_width, viewport_height) = VIEWPORT_SIZE.lock().unwrap().clone();

        let (x, y) = position;
        let (width, height) = dimensions;

        let lower_x = -width / 2.0;
        let upper_x = width / 2.0;
        let lower_y = -height / 2.0;
        let upper_y = height / 2.0;

        let lower_left = na::Vector3::new(lower_x, lower_y, 1.0);
        let lower_right = na::Vector3::new(upper_x, lower_y, 1.0);
        let upper_left = na::Vector3::new(lower_x, upper_y, 1.0);
        let upper_right = na::Vector3::new(upper_x, upper_y, 1.0);

        let flip_matrix = flip_matrix(self.flipped_horizontally, self.flipped_vertically);
        let viewport_matrix = viewport_matrix(viewport_width as f32, viewport_height as f32)
            .try_inverse()
            .expect("unreachable");
        let rotation_matrix = rotation_matrix(self.rotation);
        let translation_matrix = translation_matrix(x, y);
        let scale_matrix = scale_matrix(self.scale, self.scale);

        let transform =
            viewport_matrix * translation_matrix * flip_matrix * rotation_matrix * scale_matrix;

        let (t_lower_x, t_upper_x, t_lower_y, t_upper_y) = self.create_vertexes_for_position(
            dimensions,
            self.texture_size,
            self.texture_position,
            1.0,
            0.0,
            1.0,
        );

        let buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            device,
            BufferUsage::all(),
            [
                Vertex {
                    position: (transform * lower_left)
                        .as_slice()
                        .try_into()
                        .expect("unreached"),
                    texture: [t_lower_x, t_lower_y],
                },
                Vertex {
                    position: (transform * lower_right)
                        .as_slice()
                        .try_into()
                        .expect("unreached"),
                    texture: [t_upper_x, t_lower_y],
                },
                Vertex {
                    position: (transform * upper_left)
                        .as_slice()
                        .try_into()
                        .expect("unreached"),
                    texture: [t_lower_y, t_upper_x],
                },
                Vertex {
                    position: (transform * upper_right)
                        .as_slice()
                        .try_into()
                        .expect("unreached"),
                    texture: [t_upper_x, t_upper_y],
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
            Arc<dyn DescriptorSet + Send + Sync>,
            Box<dyn GpuFuture + Send + Sync>,
        ),
        Box<dyn Error>,
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
        (new_x, new_y): (f32, f32),
        device: Arc<Device>,
    ) -> Result<(), Box<dyn Error>> {
        let (width, height) = *VIEWPORT_SIZE.lock().unwrap();
        match self.position {
            Some((old_x, old_y)) => {
                self.position = Some((new_x, new_y));
                let (delta_x, delta_y) = (
                    (new_x - old_x) / width as f32,
                    (new_y - old_y) / height as f32,
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

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    texture: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, texture);

impl Component for Graphics {
    type Storage = VecStorage<Self>;
}

pub struct GraphicsSystem {
    recreate_swapchain: Arc<AtomicBool>,
    previous_frame_end: Box<dyn GpuFuture + Send + Sync>,
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
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    descriptor_pool: Arc<
        Mutex<FixedSizeDescriptorSetsPool<Arc<dyn PipelineLayoutAbstract + Send + Sync + 'static>>>,
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
    ) -> Result<GraphicsSystem, Box<dyn std::error::Error>> {
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
                        render_pass.clone() as Arc<dyn RenderPassAbstract + Send + Sync>,
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
            pipeline.clone() as Arc<dyn PipelineLayoutAbstract + Send + Sync>,
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
    ) -> Result<(AutoCommandBuffer, Box<dyn GpuFuture + Send + Sync>), Box<dyn std::error::Error>>
    {
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
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
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
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture;
layout(location = 0) out vec2 tex_coords;
void main() {
    gl_Position = vec4(position[0], position[1], 0.0, 1.0);
    tex_coords = texture;
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
