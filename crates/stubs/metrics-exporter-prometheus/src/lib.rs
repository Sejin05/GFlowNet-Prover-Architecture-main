use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use parking_lot::RwLock;

pub use metrics;

static GLOBAL_RECORDER: OnceLock<Arc<PrometheusRecorder>> = OnceLock::new();

fn global_recorder() -> Arc<PrometheusRecorder> {
    GLOBAL_RECORDER
        .get_or_init(|| {
            let rec = Arc::new(PrometheusRecorder::default());
            let _ = metrics::set_global_recorder(Box::leak(Box::new(PrometheusRecorderHandle(
                rec.clone(),
            ))));
            rec
        })
        .clone()
}

#[derive(Default, Clone)]
pub struct PrometheusBuilder {
    listen_addr: Option<SocketAddr>,
}

impl PrometheusBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_http_listener(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = Some(addr);
        self
    }

    pub fn install_recorder(self) -> Result<PrometheusHandle, Box<dyn std::error::Error + Send + Sync>> {
        let recorder = global_recorder();
        let handle = PrometheusHandle {
            recorder: recorder.clone(),
        };

        if let Some(addr) = self.listen_addr {
            let h = handle.clone();
            tokio::spawn(async move {
                if let Ok(listener) = tokio::net::TcpListener::bind(addr).await {
                    loop {
                        if let Ok((mut stream, _)) = listener.accept().await {
                            tokio::spawn({
                                let h_conn = h.clone();
                                async move {
                                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                                    let mut req_buf = [0u8; 1024];
                                    let _ = tokio::time::timeout(
                                        std::time::Duration::from_millis(250),
                                        stream.read(&mut req_buf),
                                    )
                                    .await;
                                    let rendered = h_conn.render();
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                        rendered.len(),
                                        rendered
                                    );
                                    let _ = stream.write_all(response.as_bytes()).await;
                                    let _ = stream.shutdown().await;
                                }
                            });
                        }
                    }
                }
            });
        }

        Ok(handle)
    }

    pub fn build_recorder(&self) -> Result<PrometheusRecorder, Box<dyn std::error::Error + Send + Sync>> {
        Ok(PrometheusRecorder::default())
    }
}

#[derive(Default)]
pub struct PrometheusRecorder {
    counters: RwLock<BTreeMap<String, u64>>,
    gauges: RwLock<BTreeMap<String, f64>>,
    histograms: RwLock<BTreeMap<String, Vec<f64>>>,
}

impl PrometheusRecorder {
    pub fn record_counter(&self, key: &str, value: u64) {
        let mut counters = self.counters.write();
        let entry = counters.entry(key.to_string()).or_insert(0);
        *entry += value;
    }

    pub fn record_gauge(&self, key: &str, value: f64) {
        let mut gauges = self.gauges.write();
        gauges.insert(key.to_string(), value);
    }

    pub fn record_histogram(&self, key: &str, value: f64) {
        let mut histograms = self.histograms.write();
        histograms.entry(key.to_string()).or_default().push(value);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        {
            let counters = self.counters.read();
            for (key, val) in counters.iter() {
                out.push_str(&format!("{key} {val}\n"));
            }
        }
        {
            let gauges = self.gauges.read();
            for (key, val) in gauges.iter() {
                out.push_str(&format!("{key} {val}\n"));
            }
        }
        {
            let histograms = self.histograms.read();
            for (key, vals) in histograms.iter() {
                let count = vals.len();
                let sum: f64 = vals.iter().sum();
                out.push_str(&format!("{key}_count {count}\n"));
                out.push_str(&format!("{key}_sum {sum}\n"));
            }
        }
        out
    }
}

struct PrometheusRecorderHandle(Arc<PrometheusRecorder>);

struct CounterHandle {
    key: String,
    recorder: Arc<PrometheusRecorder>,
}

impl metrics::CounterFn for CounterHandle {
    fn increment(&self, value: u64) {
        self.recorder.record_counter(&self.key, value);
    }

    fn absolute(&self, value: u64) {
        let mut counters = self.recorder.counters.write();
        counters.insert(self.key.clone(), value);
    }
}

struct GaugeHandle {
    key: String,
    recorder: Arc<PrometheusRecorder>,
}

impl metrics::GaugeFn for GaugeHandle {
    fn increment(&self, value: f64) {
        let mut gauges = self.recorder.gauges.write();
        let entry = gauges.entry(self.key.clone()).or_insert(0.0);
        *entry += value;
    }

    fn decrement(&self, value: f64) {
        let mut gauges = self.recorder.gauges.write();
        let entry = gauges.entry(self.key.clone()).or_insert(0.0);
        *entry -= value;
    }

    fn set(&self, value: f64) {
        self.recorder.record_gauge(&self.key, value);
    }
}

struct HistogramHandle {
    key: String,
    recorder: Arc<PrometheusRecorder>,
}

impl metrics::HistogramFn for HistogramHandle {
    fn record(&self, value: f64) {
        self.recorder.record_histogram(&self.key, value);
    }
}

impl metrics::Recorder for PrometheusRecorderHandle {
    fn describe_counter(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) {}
    fn describe_gauge(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) {}
    fn describe_histogram(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) {}

    fn register_counter(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> metrics::Counter {
        let key_str = format_key(key);
        metrics::Counter::from_arc(Arc::new(CounterHandle {
            key: key_str,
            recorder: self.0.clone(),
        }))
    }

    fn register_gauge(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> metrics::Gauge {
        let key_str = format_key(key);
        metrics::Gauge::from_arc(Arc::new(GaugeHandle {
            key: key_str,
            recorder: self.0.clone(),
        }))
    }

    fn register_histogram(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> metrics::Histogram {
        let key_str = format_key(key);
        metrics::Histogram::from_arc(Arc::new(HistogramHandle {
            key: key_str,
            recorder: self.0.clone(),
        }))
    }
}

fn format_key(key: &metrics::Key) -> String {
    let name = key.name();
    let labels: Vec<String> = key.labels().map(|l| format!("{}=\"{}\"", l.key(), l.value())).collect();
    if labels.is_empty() {
        name.to_string()
    } else {
        format!("{}{{{}}}", name, labels.join(","))
    }
}

#[derive(Clone)]
pub struct PrometheusHandle {
    recorder: Arc<PrometheusRecorder>,
}

impl PrometheusHandle {
    pub fn render(&self) -> String {
        self.recorder.render()
    }
}
