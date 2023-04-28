use std::pin::Pin;
use std::sync::Arc;

use futures::{stream::StreamExt, Stream};
use reqwest::header::HeaderMap;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    edit::Edits,
    error::{map_deserialization_error, OpenAIError, WrappedError},
    file::Files,
    image::Images,
    moderation::Moderations,
    Audio, Chat, Completions, Embeddings, FineTunes, Models,
};

#[derive(Debug, Clone)]
/// Client is a container for api key, base url, organization id, and backoff
/// configuration used to make API calls.
pub struct Client {
    http_client: Arc<reqwest::Client>,
    api_key: String,
    api_base: String,
    backoff: backoff::ExponentialBackoff,
    headers: HeaderMap,
}

/// Default v1 API base url
pub const API_BASE: &str = "https://api.openai.com/v1/";
/// Name for organization header
pub const ORGANIZATION_HEADER: &str = "OpenAI-Organization";

impl Default for Client {
    /// Create client with default [API_BASE] url and default API key from OPENAI_API_KEY env var
    fn default() -> Self {
        let mut headers = HeaderMap::new();
        let org_id = std::env::var("OPENAI_ORG_ID").ok();
        if let Some(org_id) = &org_id {
            headers.insert(ORGANIZATION_HEADER, org_id.as_str().parse().unwrap());
        }
        Self {
            http_client: Arc::new(reqwest::Client::new()),
            api_base: API_BASE.to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string()),
            backoff: Default::default(),
            headers: headers,
        }
    }
}

impl Client {
    /// Create client with default [API_BASE] url and default API key from OPENAI_API_KEY env var
    pub fn new() -> Self {
        Default::default()
    }

    /// Provide your own [client] to make HTTP requests with.
    ///
    /// [client]: reqwest::Client
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Arc::new(http_client);
        self
    }

    /// To use a different API key different from default OPENAI_API_KEY env var
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// To use a different organization id other than default
    pub fn with_org_id<S: Into<String>>(mut self, org_id: S) -> Self {
        let org_id: String = org_id.into();
        self.headers
            .insert(ORGANIZATION_HEADER, org_id.parse().unwrap());
        self
    }

    /// To use a API base url different from default [API_BASE]
    pub fn with_api_base<S: Into<String>>(mut self, api_base: S) -> Self {
        self.api_base = api_base.into();
        self
    }

    /// Exponential backoff for retrying [rate limited](https://platform.openai.com/docs/guides/rate-limits) requests.
    /// Form submissions are not retried.
    pub fn with_backoff(mut self, backoff: backoff::ExponentialBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    // pub fn api_base(&self) -> &str {
    //     &self.api_base
    // }

    fn api_base_url(&self) -> reqwest::Url {
        reqwest::Url::parse(&self.api_base).expect("Invalid API base URL")
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    // API groups

    /// To call [Models] group related APIs using this client.
    pub fn models(&self) -> Models {
        Models::new(self)
    }

    /// To call [Completions] group related APIs using this client.
    pub fn completions(&self) -> Completions {
        Completions::new(self)
    }

    /// To call [Chat] group related APIs using this client.
    pub fn chat(&self) -> Chat {
        Chat::new(self)
    }

    /// To call [Edits] group related APIs using this client.
    pub fn edits(&self) -> Edits {
        Edits::new(self)
    }

    /// To call [Images] group related APIs using this client.
    pub fn images(&self) -> Images {
        Images::new(self)
    }

    /// To call [Moderations] group related APIs using this client.
    pub fn moderations(&self) -> Moderations {
        Moderations::new(self)
    }

    /// To call [Files] group related APIs using this client.
    pub fn files(&self) -> Files {
        Files::new(self)
    }

    /// To call [FineTunes] group related APIs using this client.
    pub fn fine_tunes(&self) -> FineTunes {
        FineTunes::new(self)
    }

    /// To call [Embeddings] group related APIs using this client.
    pub fn embeddings(&self) -> Embeddings {
        Embeddings::new(self)
    }

    /// To call [Audio] group related APIs using this client.
    pub fn audio(&self) -> Audio {
        Audio::new(self)
    }

    fn build_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = self.api_base_url().join(path).expect("Invalid API path");
        println!("{:?}", url.as_str());
        self.http_client
            .request(method, url)
            .bearer_auth(self.api_key())
            .headers(self.headers.clone())
    }

    /// Make a GET request to {path} and deserialize the response body
    pub(crate) async fn get<O>(&self, path: &str) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let request = self.build_request(reqwest::Method::GET, path).build()?;
        self.execute(request).await
    }

    /// Make a DELETE request to {path} and deserialize the response body
    pub(crate) async fn delete<O>(&self, path: &str) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let request = self.build_request(reqwest::Method::DELETE, path).build()?;
        self.execute(request).await
    }

    /// Make a POST request to {path} and deserialize the response body
    pub(crate) async fn post<I, O>(&self, path: &str, request: I) -> Result<O, OpenAIError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        let request = self
            .build_request(reqwest::Method::POST, path)
            .json(&request)
            .build()?;

        self.execute(request).await
    }

    /// POST a form at {path} and deserialize the response body
    pub(crate) async fn post_form<O>(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let request = self
            .build_request(reqwest::Method::POST, path)
            .multipart(form)
            .build()?;

        self.execute(request).await
    }

    /// Deserialize response body from either error object or actual response object
    async fn process_response<O>(&self, response: reqwest::Response) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let status = response.status();
        let bytes = response.bytes().await?;

        if !status.is_success() {
            let wrapped_error: WrappedError = serde_json::from_slice(bytes.as_ref())
                .map_err(|e| map_deserialization_error(e, bytes.as_ref()))?;

            return Err(OpenAIError::ApiError(wrapped_error.error));
        }

        let response: O = serde_json::from_slice(bytes.as_ref())
            .map_err(|e| map_deserialization_error(e, bytes.as_ref()))?;
        Ok(response)
    }

    /// Execute any HTTP requests and retry on rate limit, except streaming ones as they cannot be cloned for retrying.
    async fn execute<O>(&self, request: reqwest::Request) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        // Check if the request is cloneable
        if let Some(cloned_request) = request.try_clone() {
            // Only clone-able requests can be retried
            backoff::future::retry(self.backoff.clone(), || {
                let request = cloned_request.try_clone().unwrap();
                let client = self.http_client.clone();
                async move {
                    let response = client
                        .execute(request)
                        .await
                        .map_err(OpenAIError::Reqwest)?;

                    match self.process_response(response).await {
                        Ok(response) => Ok(response),
                        Err(OpenAIError::RateLimited(error)) => Err(backoff::Error::Transient {
                            err: OpenAIError::ApiError(error),
                            retry_after: None,
                        }),
                        Err(e) => Err(backoff::Error::Permanent(e)),
                    }
                }
            })
            .await
        } else {
            // For non-cloneable requests, execute without retry logic
            let response = self.http_client.execute(request).await?;
            self.process_response(response).await
        }
    }

    /// Make HTTP POST request to receive SSE
    pub(crate) async fn post_stream<I, O>(
        &self,
        path: &str,
        request: I,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        I: Serialize,
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        let event_source = self
            .build_request(reqwest::Method::POST, path)
            .json(&request)
            .eventsource()
            .unwrap();

        Client::stream(event_source).await
    }

    /// Make HTTP GET request to receive SSE
    pub(crate) async fn get_stream<Q, O>(
        &self,
        path: &str,
        query: &Q,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        Q: Serialize + ?Sized,
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        let event_source = self
            .build_request(reqwest::Method::GET, path)
            .query(query)
            .eventsource()
            .unwrap();

        Client::stream(event_source).await
    }

    /// Request which responds with SSE.
    /// [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format)
    pub(crate) async fn stream<O>(
        mut event_source: EventSource,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            while let Some(ev) = event_source.next().await {
                match ev {
                    Err(e) => {
                        if let Err(_e) = tx.send(Err(OpenAIError::StreamError(e.to_string()))) {
                            // rx dropped
                            break;
                        }
                    }
                    Ok(event) => match event {
                        Event::Message(message) => {
                            if message.data == "[DONE]" {
                                break;
                            }

                            let response = match serde_json::from_str::<O>(&message.data) {
                                Err(e) => {
                                    Err(map_deserialization_error(e, &message.data.as_bytes()))
                                }
                                Ok(output) => Ok(output),
                            };

                            if let Err(_e) = tx.send(response) {
                                // rx dropped
                                break;
                            }
                        }
                        Event::Open => continue,
                    },
                }
            }

            event_source.close();
        });

        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }
}
