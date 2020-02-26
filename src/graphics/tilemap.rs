use image::GenericImage;
use std::collections::HashMap;

use crate::graphics::imagewrapper;

#[derive(Deserialize)]
struct TileLayer {
    // id: i32,
    // height: u32,
    // width: u32,
    // name: String,
    // #[serde(rename = "type")]
    // layer_type: String, // TODO: enum?
    // opacity: i32,
    // x: i32,
    // y: i32,
    data: Vec<u32>,
}

#[derive(Deserialize)]
pub struct Tileset {
    #[serde(rename = "firstgid")]
    first_gid: u32,

    source: String,
}

#[derive(Deserialize)]
pub struct TileMapMetadata {
    // #[serde(rename = "compressionlevel")]
    // compression_level: i32,
    height: u32,
    width: u32,
    // infinite: bool,

    // #[serde(rename = "nextlayerid")]
    // next_layer_id: i32,

    // #[serde(rename = "nextobjectid")]
    // next_object_id: i32,

    // #[serde(rename = "renderorder")]
    // render_order: String, // TODO: enum?

    // orientation: String, // TODO: enum?

    // #[serde(rename = "tiledversion")]
    // tiled_version: String,
    #[serde(rename = "tileheight")]
    tile_height: u32,

    #[serde(rename = "tilewidth")]
    tile_width: u32,

    // #[serde(rename = "type")]
    // map_type: String, // TODO: enum?

    // version: f64,
    layers: Vec<TileLayer>,

    tilesets: Vec<Tileset>,
}

pub struct TileMap {
    pub metadata: TileMapMetadata,
}

impl TileMap {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<TileMap, Box<dyn std::error::Error>> {
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(TileMap {
            metadata: serde_json::from_reader(file)?,
        })
    }
}

impl TileMapMetadata {
    pub fn build_image(self) -> Result<imagewrapper::Image, Box<dyn std::error::Error>> {
        let mut tileset_cache = HashMap::new();
        let mut buffer =
            image::ImageBuffer::new(self.height * self.tile_height, self.width * self.tile_width);

        for layer in &self.layers {
            self.render_layer(layer, &mut buffer, &mut tileset_cache)?;
        }

        Ok(imagewrapper::Image::wrap(image::DynamicImage::ImageRgba8(
            buffer,
        )))
    }

    fn render_layer<T: GenericImage<Pixel = image::Rgba<u8>>>(
        &self,
        layer: &TileLayer,
        buffer: &mut T,
        tileset_cache: &mut HashMap<u32, image::DynamicImage>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.height {
            for j in 0..self.width {
                let tile =
                    self.get_tile(layer.data[(i * self.width + j) as usize], tileset_cache)?;
                buffer.copy_from(&tile, j * self.tile_width, i * self.tile_height)?;
            }
        }

        Ok(())
    }

    fn get_tile(
        &self,
        tile_gid: u32,
        tileset_cache: &mut HashMap<u32, image::DynamicImage>,
    ) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {
        if tileset_cache.contains_key(&tile_gid) {
            return Ok(tileset_cache.get(&tile_gid).unwrap().clone());
        }

        let mut selected_tileset: Option<&Tileset> = None;

        for tileset in &self.tilesets {
            if tileset.first_gid > tile_gid {
                break;
            }
            selected_tileset = Some(tileset);
        }

        selected_tileset
            .ok_or(Box::new(crate::error::NoneError) as Box<dyn std::error::Error>)
            .and_then(|tileset| {
                let mut image = imagewrapper::Image::load(&tileset.source)?;
                image
                    .tiles(self.tile_width, self.tile_height)
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, tile)| {
                        tileset_cache.insert(i as u32 + tileset.first_gid, tile.into_raw());
                    });

                Ok(tileset_cache.get(&tile_gid).unwrap().clone()) // should always be present
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize() {
        match TileMap::from_file("resources/testmap.json") {
            Ok(_) => (),
            Err(e) => assert!(false, "Error deserializing tilemap: {}", e),
        }
    }
}
