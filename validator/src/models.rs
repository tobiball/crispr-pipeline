use bio_seq::{codec::dna::Dna};
use polars::error::PolarsResult;
use polars::frame::DataFrame;
use polars::prelude::{col, lit, DataFrameJoinOps, IntoLazy, JoinArgs, JoinType, PolarsError};
use tracing::{debug, error};

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
    "strand",
    "efficacy",
    "start",
    "end"
];


pub trait Dataset {
    /// Loads and preprocesses the dataset, returning a DataFrame.
    fn load(&self) -> PolarsResult<DataFrame>;

    fn validate_columns(df: &DataFrame, dataset_name: &str) -> PolarsResult<()> {
        let df_columns = df.get_column_names();
        let missing: Vec<_> = REQUIRED_COLUMNS
            .iter()
            .filter(|&&col| !df_columns.iter().any(|df_col| df_col.as_str() == col))
            .collect();

        if !missing.is_empty() {
            // Log the problem:
            error!("Dataset {} is missing required columns: {:?}", dataset_name, missing);

            // Return a PolarsError so your calling code can handle it:
            return Err(PolarsError::ComputeError(
                format!("Validation check! Missing required columns: {:?}", missing).into(),
            ));
        }
        Ok(())
    }

    fn filter_for_ceg_only(df: DataFrame, df_ceg: DataFrame) -> PolarsResult<DataFrame> {

        debug!("df_ceg columns: {:?}", df_ceg.get_column_names());

        // 1) Extract the GENE column as a Series
        let df_filtered = df.join(&df_ceg, ["Gene"], ["GENE"], JoinArgs::from(JoinType::Semi))?;

        debug!("{:?}",df_filtered);


        Ok(df_filtered)
    }
    /// A convenience method that loads and validates in one go
    fn load_validated(&self, dataset_name: &str, cegs: DataFrame) -> PolarsResult<DataFrame> {
        let df = self.load()?;
        Self::validate_columns(&df, dataset_name)?;
        let df_filtered = Self::filter_for_ceg_only(df,cegs)?;
        Ok(df_filtered)
    }
    }

