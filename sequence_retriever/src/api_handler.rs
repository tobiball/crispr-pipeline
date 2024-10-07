// src/api_handler.rs

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use serde_json::Value;
use std::error::Error;
use std::thread;
use std::time::Duration;

pub struct APIHandler {
    client: Client,
    base_url: String,
}

impl APIHandler {
    pub fn new(base_url: &str) -> Result<Self, Box<dyn Error>> {
        let mut headers = HeaderMap::new();
        headers.insert("Accept", HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static("YourAppName/1.0"));

        let client = Client::builder()
            .default_headers(headers)
            .build()?;

        Ok(Self {
            client,
            base_url: base_url.to_string(),
        })
    }

    pub fn get(&self, endpoint: &str) -> Result<Value, Box<dyn Error>> {
        let url = format!("{}{}", self.base_url, endpoint);
        println!("{}", url);
        self.make_request_with_retry(&url, 3)
    }

    pub fn get_plain_text(&self, endpoint: &str) -> Result<String, Box<dyn Error>> {
        let url = format!("{}{}", self.base_url, endpoint);
        self.make_plain_text_request_with_retry(&url, 3)
    }

    fn make_request_with_retry(&self, url: &str, max_attempts: u32) -> Result<Value, Box<dyn Error>> {
        let mut attempts = 0;

        loop {
            let response = self.client.get(url).send()?;

            if response.status().is_success() {
                return Ok(response.json()?);
            } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                attempts += 1;
                if attempts >= max_attempts {
                    return Err(format!("Exceeded maximum retries for URL: {}", url).into());
                }

                if let Some(retry_after) = response.headers().get("Retry-After") {
                    let wait_time = retry_after.to_str()?.parse::<u64>().unwrap_or(1);
                    eprintln!("Rate limited. Waiting {} seconds before retrying...", wait_time);
                    thread::sleep(Duration::from_secs(wait_time));
                } else {
                    eprintln!("Rate limited. Waiting 1 second before retrying...");
                    thread::sleep(Duration::from_secs(1));
                }
            } else {
                let status = response.status();
                let error_text = response.text()?;
                return Err(format!(
                    "Failed to fetch data from URL: {}. Status: {}. Error: {}",
                    url, status, error_text
                ).into());
            }
        }
    }

    fn make_plain_text_request_with_retry(&self, url: &str, max_attempts: u32) -> Result<String, Box<dyn Error>> {
        let mut attempts = 0;

        loop {
            let response = self.client.get(url)
                .header("Accept", "text/plain")
                .send()?;

            if response.status().is_success() {
                return Ok(response.text()?);
            } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                attempts += 1;
                if attempts >= max_attempts {
                    return Err(format!("Exceeded maximum retries for URL: {}", url).into());
                }

                if let Some(retry_after) = response.headers().get("Retry-After") {
                    let wait_time = retry_after.to_str()?.parse::<u64>().unwrap_or(1);
                    eprintln!("Rate limited. Waiting {} seconds before retrying...", wait_time);
                    thread::sleep(Duration::from_secs(wait_time));
                } else {
                    eprintln!("Rate limited. Waiting 1 second before retrying...");
                    thread::sleep(Duration::from_secs(1));
                }
            } else {
                let status = response.status();
                let error_text = response.text()?;
                return Err(format!(
                    "Failed to fetch data from URL: {}. Status: {}. Error: {}",
                    url, status, error_text
                ).into());
            }
        }
    }
}
