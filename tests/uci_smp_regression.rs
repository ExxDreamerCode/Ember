use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

fn spawn_ember() -> (Child, Receiver<String>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ember"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn Ember UCI process");
    let stdout = child.stdout.take().expect("capture Ember stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    (child, rx)
}

fn wait_for_line(rx: &Receiver<String>, prefix: &str, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(remaining) {
            Ok(line) if line.starts_with(prefix) => return Some(line),
            Ok(_) => {}
            Err(_) => return None,
        }
    }
    None
}

#[test]
fn immediate_stop_interrupts_lazy_smp_search() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Threads value 4").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some(),
        "Ember did not finish UCI initialization"
    );

    writeln!(stdin, "position startpos").unwrap();
    write!(stdin, "go infinite\nstop\n").unwrap();
    stdin.flush().unwrap();

    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(2));
    if bestmove.is_none() {
        let _ = child.kill();
        let _ = child.wait();
        panic!("an immediate UCI stop was lost by the Lazy SMP search");
    }

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("wait for Ember UCI process");
    assert!(status.success(), "Ember exited with {status}");
}
