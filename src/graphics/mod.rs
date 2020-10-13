use std::convert::TryInto;
use std::sync::{Arc, Mutex};

use specs::prelude::*;

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

use winit::window::Window;

use crate::core::{GameObjectComponent, MethodAdder, Scriptable};
use crate::error::NoneError;
use crate::position::Position;

pub mod anim;
mod draw2d;
mod imagewrapper;
pub mod sprite;
pub mod tilemap;
pub mod window;

lazy_static! {
    static ref VIEWPORT_SIZE: Mutex<(u32, u32)> = Mutex::new((0, 0));
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

fn rotation_matrix(degrees: f32) -> na::Matrix3<f32> {
    na::Matrix3::new(
        degrees.to_radians().cos(),
        degrees.to_radians().sin(),
        0.0,
        -degrees.to_radians().sin(),
        degrees.to_radians().cos(),
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
    image: imagewrapper::Image,
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
    image: Option<ImageDesc>,
    flipped_horizontally: bool,
    flipped_vertically: bool,
}

enum ImageDesc {
    Image(imagewrapper::Image),
    ImagePath(std::path::PathBuf, image::ImageFormat),
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

    pub fn image(mut self, image: imagewrapper::Image) -> Self {
        self.image = Some(ImageDesc::Image(image));
        self
    }

    pub fn load_image<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
        format: image::ImageFormat,
    ) -> Self {
        self.image = Some(ImageDesc::ImagePath(path.as_ref().to_path_buf(), format));
        self
    }

    pub fn build(self) -> Result<Graphics, Box<dyn std::error::Error>> {
        let image = self
            .image
            .map(|x| match x {
                ImageDesc::Image(image) => Ok(image),
                ImageDesc::ImagePath(path, format) => {
                    imagewrapper::Image::load_with_ext(path, format)
                }
            })
            .transpose()?;

        let size = image
            .as_ref()
            .map(|x| x.dimensions())
            .map(|(width, height)| (width as f32, height as f32));

        Ok(Graphics {
            image: image.ok_or(NoneError)?,
            position: self.position,
            scale: self.scale.unwrap_or(1.0),
            texture_position: self.texture_position.unwrap_or((0.0, 0.0)),
            texture_size: self.texture_size.or(size).ok_or(NoneError)?,
            texture_buffer: None,
            vertex_buffer: None,
            rotation: self.rotation.unwrap_or(0.0),
            flipped_horizontally: self.flipped_horizontally,
            flipped_vertically: self.flipped_vertically,
        })
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
    ) -> Result<Arc<CpuAccessibleBuffer<[Vertex]>>, Box<dyn std::error::Error>> {
        let reg_dimensions = self.image.dimensions();
        let dimensions = (reg_dimensions.0 as f32, reg_dimensions.1 as f32);
        let (viewport_width, viewport_height) = VIEWPORT_SIZE.lock()?.clone();

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
            false,
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
        Box<dyn std::error::Error>,
    > {
        let dimensions = self.image.dimensions();
        let image_dimensions = (dimensions.0 as f64, dimensions.1 as f64);

        let image_data = self.image.data();
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
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (width, height) = *VIEWPORT_SIZE.lock()?;
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
    window_event_sender: std::sync::mpsc::Sender<crate::windowing::WindowEvent>,
    window_event_reciever: std::sync::mpsc::Receiver<crate::windowing::WindowEvent>,
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
    descriptor_pool: Arc<Mutex<FixedSizeDescriptorSetsPool>>,
    sampler: Arc<Sampler>,
}

#[derive(Debug)]
struct NoWindowError;

impl std::fmt::Display for NoWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "No open window found")
    }
}

impl std::error::Error for NoWindowError {}

impl GraphicsSystem {
    pub fn new(
        windowing: &mut crate::windowing::WindowSystem,
    ) -> Result<GraphicsSystem, Box<dyn std::error::Error>> {
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

        let surface = windowing.surface(instance.clone())?;
        let window = surface.window();

        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .ok_or(NoneError)?;

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

        let size = surface.window().inner_size();
        let new_width = size.width;
        let new_height = size.height;

        *VIEWPORT_SIZE.lock()? = (new_width, new_height);

        let (swapchain, images) = {
            let caps = surface.capabilities(physical)?;

            let usage = caps.supported_usage_flags;
            let alpha = caps
                .supported_composite_alpha
                .iter()
                .next()
                .ok_or(NoneError)?;
            let format = caps.supported_formats[0].0;

            let size = window.inner_size();

            let initial_dimensions = [size.width, size.height];

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
                vulkano::swapchain::FullscreenExclusive::Default,
                true,
                vulkano::swapchain::ColorSpace::SrgbNonLinear,
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

        let mut dynamic_state = DynamicState::none();
        let framebuffers =
            window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state)?;

        let layout = pipeline.descriptor_set_layout(0).ok_or(NoneError)?.clone();

        let descriptor_pool = Arc::new(Mutex::new(FixedSizeDescriptorSetsPool::new(layout)));

        Ok(GraphicsSystem {
            window_event_sender: windowing.window_event_sender(),
            window_event_reciever: windowing.window_event_reciever().ok_or(NoneError)?,
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
            let position = data.position.ok_or(NoneError)?;
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

    fn do_run(
        &mut self,
        (graphics, positions): <Self as System>::SystemData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let window = self.surface.window();
        self.previous_frame_end.cleanup_finished();
        while let Ok(ev) = self.window_event_reciever.try_recv() {
            match ev {
                crate::windowing::WindowEvent::Resized => {
                    let size = window.inner_size();
                    let dimensions = (size.width, size.height);
                    *VIEWPORT_SIZE.lock()? = dimensions;

                    let (new_swapchain, new_images) = self
                        .swapchain
                        .recreate_with_dimensions([size.width, size.height])?;

                    self.swapchain = new_swapchain;
                    self.framebuffers = window_size_dependent_setup(
                        &new_images,
                        self.render_pass.clone(),
                        &mut self.dynamic_state,
                    )?;
                }
            }
        }

        let (image_num, recreate_swapchain_needed, future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.window_event_sender
                        .send(crate::windowing::WindowEvent::Resized)?;
                    return Ok(());
                }
                Err(err) => return Err(Box::new(err)),
            };

        if recreate_swapchain_needed {
            self.window_event_sender
                .send(crate::windowing::WindowEvent::Resized)?;
        }

        let (cb, previous_frame_end) = self.build_render_pass(positions, graphics, image_num)?;

        let future = previous_frame_end
            .join(future)
            .then_execute(self.queue.clone(), cb)?
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        Ok(match future {
            Ok(future) => {
                if cfg!(target_os = "macos") {
                    // Workaround for moltenvk issue (hang on close)
                    // FIXME Remove once motenvk is fixed
                    future.wait(None).expect("waiting on fence failed");
                }
                self.previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                self.window_event_sender
                    .send(crate::windowing::WindowEvent::Resized)?;
                self.previous_frame_end = Box::new(sync::now(self.device.clone())) as Box<_>;
            }
            Err(e) => {
                error!("{:?}", e);
                self.previous_frame_end = Box::new(sync::now(self.device.clone())) as Box<_>;
            }
        })
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

    fn run(&mut self, data: Self::SystemData) {
        match self.do_run(data) {
            Ok(()) => (),
            Err(e) => error!("Error in graphics system: {:?}", e),
        }
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Result<Vec<Arc<dyn FramebufferAbstract + Send + Sync>>, Box<dyn std::error::Error>> {
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
            Ok(Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())?
                    .build()?,
            ) as Arc<dyn FramebufferAbstract + Send + Sync>)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
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
