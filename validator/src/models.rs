use bio_seq::{codec::dna::Dna};
use polars::error::PolarsResult;
use polars::frame::DataFrame;
use polars::prelude::PolarsError;

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

pub const REQUIRED_COLUMNS: &[&str] = &[
    "chromosome",
    "sgRNA",
    "sequence",
    "position",
    "strand",
    "efficiency_score",
];


pub trait Dataset {
    /// Loads and preprocesses the dataset, returning a DataFrame.
    fn load(&self) -> PolarsResult<DataFrame>;

    fn validate_columns(df:&DataFrame) -> PolarsResult<()> {
            let df_columns = df.get_column_names(); // Vec<&PlSmallStr>
            let missing: Vec<_> = REQUIRED_COLUMNS
                .iter()
                .filter(|&&col| !df_columns.iter().any(|df_col| df_col.as_str() == col))
                .collect();

            if !missing.is_empty() {
                return Err(PolarsError::ComputeError(
                    format!("Missing required columns: {:?}", missing).into(),
                ));
            }
            Ok(())
        }
    /// A convenience method that loads and validates in one go
    fn load_validated(&self) -> PolarsResult<DataFrame> {
        let df = self.load()?;
        Self::validate_columns(&df)?;
        Ok(df)
    }
    }

