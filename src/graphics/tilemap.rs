#[derive(Deserialize)]
struct TileLayer {
    id: i32,
    height: usize,
    width: usize,
    name: String,
    #[serde(rename = "type")]
    layer_type: String, // TODO: enum?
    opacity: i32,
    x: i32,
    y: i32,

    data: Vec<usize>
}

#[derive(Deserialize)]
pub struct Tileset {
    #[serde(rename = "firstgid")]
    first_gid: usize,

    source: String
}

#[derive(Deserialize)]
pub struct TileMap {
    #[serde(rename = "compressionlevel")]
    compression_level: i32,

    height: usize,
    width: usize,
    infinite: bool,

    #[serde(rename = "nextlayerid")]
    next_layer_id: i32,

    #[serde(rename = "nextobjectid")]
    next_object_id: i32,

    #[serde(rename = "renderorder")]
    render_order: String, // TODO: enum?

    orientation: String, // TODO: enum?

    #[serde(rename = "tiledversion")]
    tiled_version: String,

    #[serde(rename = "tileheight")]
    tile_height: usize,

    #[serde(rename = "tilewidth")]
    tile_width: usize,

    #[serde(rename = "type")]
    map_type: String, // TODO: enum?

    version: f64,

    layers: Vec<TileLayer>

}

impl TileMap {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<TileMap, Box<dyn std::error::Error>> {
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(serde_json::from_reader(file)?)
    }

    pub fn build_image(self) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize() {
        match TileMap::from_file("resources/testmap.json") {
            Ok(_) => (),
            Err(e) => assert!(false, "Error deserializing tilemap: {}", e)
        }
    }
}