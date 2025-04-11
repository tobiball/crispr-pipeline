use polars::error::PolarsResult;
use polars::frame::DataFrame;
use crate::helper_functions::read_csv;
use crate::models::Dataset;

pub struct AnyDataset {
    pub path: String,
}


impl Dataset for AnyDataset  {
    fn load(&self) -> PolarsResult<DataFrame> {
        read_csv(&self.path)
    }

    fn augment_guides(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok(df)
    }

    fn mageck_efficency_scoring(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok(df)
    }
}

