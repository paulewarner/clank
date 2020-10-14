use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct WindowBuilderSystemConfig {
    default_font_path: String,
    default_color: (u8, u8, u8),
    default_font_size: f32,
}

pub struct WindowBuilderSystem {
    default_font: Arc<rusttype::Font<'static>>,
    default_font_size: f32,
    default_color: (u8, u8, u8),
}

impl WindowBuilderSystem {
    pub fn new(config: WindowBuilderSystemConfig) -> Result<WindowBuilderSystem, Box<dyn Error>> {
        Ok(WindowBuilderSystem {
            default_font: Arc::new(load_font(config.default_font_path)?),
            default_color: config.default_color,
            default_font_size: config.default_font_size,
        })
    }

    pub fn create_window(&self) -> WindowBuilder {
        WindowBuilder {
            text: "".to_owned(),
            color: self.default_color,
            size: self.default_font_size,
            font: self.default_font.clone(),
        }
    }
}

pub struct WindowBuilder {
    text: String,
    color: (u8, u8, u8),
    size: f32,
    font: Arc<rusttype::Font<'static>>,
}

impl WindowBuilder {
    pub fn text<R: AsRef<str>>(mut self, text: R) -> Self {
        self.text = String::from(text.as_ref());
        self
    }

    pub fn font(mut self, f: Arc<rusttype::Font<'static>>) -> Self {
        self.font = f;
        self
    }

    pub fn color<P: AsRef<Path>>(mut self, color: (u8, u8, u8)) -> Self {
        self.color = color;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn build(self) -> Result<crate::graphics::Graphics, Box<dyn Error>> {
        let raw = layout_text(self.text, self.size, self.font.as_ref(), self.color)?;
        let image = crate::graphics::imagewrapper::Image::wrap(raw);
        crate::graphics::Graphics::new().image(image).build()
    }
}

fn load_font<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<rusttype::Font<'static>, Box<dyn std::error::Error>> {
    let mut font_data = Vec::new();
    let mut reader = BufReader::new(File::open(path)?);
    reader.read_to_end(&mut font_data)?;
    Ok(rusttype::Font::try_from_vec(font_data).ok_or(crate::error::NoneError)?)
}

fn layout_text(
    text: String,
    size: f32,
    font: &rusttype::Font<'static>,
    (r, g, b): (u8, u8, u8),
) -> Result<image::DynamicImage, Box<dyn Error>> {
    let scale = rusttype::Scale::uniform(size);
    let v_metrics = font.v_metrics(scale);

    let glyphs: Vec<_> = font
        .layout(
            text.as_ref(),
            scale,
            rusttype::point(20.0, 20.0 + v_metrics.ascent),
        )
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

    let mut image = image::DynamicImage::new_rgba8(glyphs_width + 40, glyphs_height + 40).to_rgba();

    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            // Draw the glyph into the image per-pixel by using the draw closure
            glyph.draw(|x, y, v| {
                image.put_pixel(
                    // Offset the position by the glyph bounding box
                    x + bounding_box.min.x as u32,
                    y + bounding_box.min.y as u32,
                    // Turn the coverage into an alpha value
                    image::Rgba([r, g, b, (v * 255.0) as u8]),
                )
            });
        }
    }

    Ok(image::DynamicImage::ImageRgba8(image))
}
