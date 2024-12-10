use bio_seq::{codec::dna::Dna};
use polars::error::PolarsResult;
use polars::frame::DataFrame;




pub enum ValidationData {
    Avana(DataFrame)
}


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

pub struct PredictionGuide {
    pub tool: ScreenLibrary,
    pub guide:Dna,
    pub efficiency: u8,
}

pub enum PredictionTool {
    CHOPCHOP
}



impl ValidationData {
    pub fn extract_dataframe(self) -> DataFrame {
        match self { ValidationData::Avana(avana_dataset) => avana_dataset }
    }
}

/// Trait representing a dataset in the CRISPR pipeline.
pub trait Dataset {
    /// Loads and preprocesses the dataset, returning a DataFrame.
    fn load(&self) -> PolarsResult<DataFrame>;
}