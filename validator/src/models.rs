use bio_seq::{codec::dna::Dna};


pub struct ScreenGuide {
    pub data_source: DataSource,
    pub library: ScreenLibrary,
    pub guide:Dna,
    pub efficiency: u8,
}

pub struct SeqGuide {

}

pub enum DataSource {
    DepMap
}

pub enum ScreenLibrary {
    Avana
}

pub struct PredicitonTool {
    pub tool: ScreenLibrary,
    pub guide:Dna,
    pub efficiency: u8,
}

pub enum PredictionTool {
    CHOPCHOP
}