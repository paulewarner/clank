use std::fs::File;
use std::io::{BufReader, Read};

use image::GenericImageView;

use rusttype::{point, Font, Scale};

use crate::error::NoneError;

pub type ImageFormat = image::ImageFormat;

#[derive(Clone)]
pub struct Image {
    raw: image::DynamicImage,
}

impl Image {
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        format: ImageFormat,
    ) -> Result<Image, Box<dyn std::error::Error>> {
        let raw = image::load(BufReader::new(File::open(path)?), format)?;
        Ok(Image { raw })
    }

    pub fn load_text<P: AsRef<std::path::Path>>(
        text: String,
        path: P,
        color: (u8, u8, u8),
        size: f32,
    ) -> Result<Image, Box<dyn std::error::Error>> {
        let font = load_font(path)?;
        Image::text(text, font, color, size)
    }

    pub fn text(
        text: String,
        font: Font,
        color: (u8, u8, u8),
        size: f32,
    ) -> Result<Image, Box<dyn std::error::Error>> {
        let scale = Scale::uniform(size);
        let v_metrics = font.v_metrics(scale);
        let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;

        let glyphs: Vec<_> = text
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
                .and_then(|g| Some(g.pixel_bounding_box()?.max.x))
                .ok_or(NoneError)?;
            let max_x = glyphs
                .last()
                .and_then(|g| Some(g.pixel_bounding_box()?.max.x))
                .ok_or(NoneError)?;
            (max_x - min_x) as u32
        };

        let glyphs_height = {
            let min_y = glyphs
                .first()
                .and_then(|g| Some(g.pixel_bounding_box()?.max.y))
                .ok_or(NoneError)?;
            let max_y = glyphs
                .last()
                .and_then(|g| Some(g.pixel_bounding_box()?.max.y))
                .ok_or(NoneError)?;
            (max_y - min_y) as u32
        };

        let mut image =
            image::DynamicImage::new_rgba8(glyphs_width + 40, glyphs_height + 40).to_rgba();

        for glyph in glyphs {
            if let Some(bounding_box) = glyph.pixel_bounding_box() {
                // Draw the glyph into the image per-pixel by using the draw closure
                glyph.draw(|x, y, v| {
                    image.put_pixel(
                        // Offset the position by the glyph bounding box
                        x + bounding_box.min.x as u32,
                        y + bounding_box.min.y as u32,
                        // Turn the coverage into an alpha value
                        image::Rgba([color.0, color.1, color.2, (v * 255.0) as u8]),
                    )
                });
            }
        }

        Ok(Image {
            raw: image::DynamicImage::ImageRgba8(image),
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.raw.dimensions()
    }

    pub fn data(&self) -> Vec<u8> {
        self.raw.clone().to_rgba().into_raw()
    }

    pub fn get_image_by_index(&mut self, (width, height): (u32, u32), index: u32) -> Image {
        let (s_width, _) = self.dimensions();
        let sheet_width = s_width / width;
        let (x, y) = (
            (index % sheet_width) * width,
            (index / sheet_width) * height,
        );
        Image {
            raw: self.raw.crop(x, y, width, height),
        }
    }
}

fn load_font<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Font<'static>, Box<dyn std::error::Error>> {
    let mut font_data = Vec::new();
    load_file(path)?.read_to_end(&mut font_data)?;
    Ok(Font::from_bytes(font_data)?)
}

fn load_file<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<BufReader<File>, Box<dyn std::error::Error>> {
    Ok(BufReader::new(File::open(path)?))
}
