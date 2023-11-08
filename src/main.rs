//! Concurrent experiments for Monte-Carlo pi estimation.

use rand::Rng;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

fn monte_carlo_sim(n: usize) -> usize {
    let mut rng = rand::thread_rng();
    let mut inside = 0;
    for _ in 0..n {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }
    return inside;
}

/// Simple (single-threaded) Monte-Carlo pi estimation.
fn single_pi(n: usize) -> f64 {
    let inside = monte_carlo_sim(n);
    return 4.0 * inside as f64 / n as f64;
}

// Fork-join parallel Monte-Carlo pi estimation.
fn fork_join_pi(n: usize) -> f64 {
    let n_workers = thread::available_parallelism().unwrap().get();
    let mut handles = vec![];
    for _ in 0..n_workers {
        let handle = thread::spawn(move || {
            let inside = monte_carlo_sim(n / n_workers);
            return inside;
        });
        handles.push(handle);
    }
    let mut inside = 0;
    for handle in handles {
        inside += handle.join().unwrap();
    }
    return 4.0 * inside as f64 / n as f64;
}

/// Message-passing parallel Monte-Carlo pi estimation.
fn message_passing_pi(n: usize) -> f64 {
    let n_workers = thread::available_parallelism().unwrap().get();
    let mut handles = vec![];
    let (tx, rx) = mpsc::channel();
    for _ in 0..n_workers {
        let tx = tx.clone();
        let handle = thread::spawn(move || {
            let inside = monte_carlo_sim(n / n_workers);
            tx.send(inside).unwrap();
        });
        handles.push(handle);
    }

    let mut inside = 0;

    for _ in 0..n_workers {
        inside += rx.recv().unwrap();
    }

    // we may want to join the threads, but it is not necessary

    return 4.0 * inside as f64 / n as f64;
}

/// Shared-memory parallel Monte-Carlo pi estimation.
/// Note how Rust prevents us to share mutable data across threads (Fearless Concurrency!)
fn shared_memory_pi(n: usize) -> f64 {
    let n_workers = thread::available_parallelism().unwrap().get();
    let shared_inside = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    for _ in 0..n_workers {
        let shared_inside = shared_inside.clone(); // shadowing
        let handle = thread::spawn(move || {
            let inside = monte_carlo_sim(n / n_workers);
            let mut shared_inside = shared_inside.lock().unwrap(); // shadowing
            *shared_inside += inside;
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let shared_inside = shared_inside.lock().unwrap(); // shadowing
    return 4.0 * *(shared_inside) as f64 / n as f64;
}

/// Continuous message-passing estimation
/// Note: a similar co continuous approach can be used with shared memory, but
/// the main thread won't get notified when the estimation improves.
fn continuous_message_passing_pi() {
    let n_chunk = 100_000;

    let (tx, rx) = mpsc::channel();
    let n_workers = thread::available_parallelism().unwrap().get();
    for _ in 0..n_workers {
        let tx = tx.clone();
        thread::spawn(move || loop {
            let inside = monte_carlo_sim(n_chunk);
            tx.send(inside).unwrap();
        });
    }
    let mut n = 0;
    let mut inside = 0;
    loop {
        inside += rx.recv().unwrap();
        n += n_chunk;
        println!("Message Passing: π = {}", 4.0 * inside as f64 / n as f64);
    }
}

fn main() {
    const N: usize = 10_000_000;

    let methods: Vec<(&str, fn(usize) -> f64)> = vec![
        ("Single Thread", single_pi),
        ("Fork & Join", fork_join_pi),
        ("Message Passing", message_passing_pi),
        ("Shared Memory", shared_memory_pi),
    ];

    for (name, f) in methods {
        let start = Instant::now();
        let pi = f(N);
        let duration = start.elapsed();

        println!("{}: π = {},  elapsed: {:?}", name, pi, duration);
    }

    continuous_message_passing_pi();
}
