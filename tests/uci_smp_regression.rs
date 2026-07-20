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

fn info_number(line: &str, field: &str) -> Option<u64> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    parts
        .windows(2)
        .find(|pair| pair[0] == field)
        .and_then(|pair| pair[1].parse().ok())
}

fn wait_for_info_time_at_least(rx: &Receiver<String>, minimum_ms: u64, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(remaining) {
            Ok(line)
                if line.starts_with("info ")
                    && info_number(&line, "time").is_some_and(|time| time >= minimum_ms) =>
            {
                return true;
            }
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    false
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

#[test]
fn completed_lazy_smp_search_emits_final_aggregate_info() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 4").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some(),
        "Ember did not finish UCI initialization"
    );

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 1").unwrap();
    stdin.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut depth_one_nodes = Vec::new();
    let mut received_bestmove = false;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(remaining) {
            Ok(line) if line.starts_with("bestmove ") => {
                received_bestmove = true;
                break;
            }
            Ok(line) if info_number(&line, "depth") == Some(1) => {
                if let Some(nodes) = info_number(&line, "nodes") {
                    depth_one_nodes.push(nodes);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("wait for Ember UCI process");

    assert!(
        received_bestmove,
        "Ember did not complete the fixed-depth search"
    );
    assert!(status.success(), "Ember exited with {status}");
    assert!(
        depth_one_nodes.len() >= 2,
        "missing final aggregate info after the worker depth report: {depth_one_nodes:?}"
    );
    assert!(
        depth_one_nodes.last() >= depth_one_nodes.first(),
        "final aggregate node count went backwards: {depth_one_nodes:?}"
    );
}

#[test]
fn completed_ponder_search_waits_for_ponderhit() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "option name Ponder ", Duration::from_secs(2)).is_some(),
        "Ember did not advertise UCI pondering"
    );
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 4").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some(),
        "Ember did not finish UCI initialization"
    );

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go ponder depth 1").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_millis(250)).is_none(),
        "a completed ponder search must not move before ponderhit"
    );

    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(2)).is_some(),
        "ponderhit did not release the completed result"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn active_ponder_search_ignores_move_time_until_ponderhit() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 4").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go ponder movetime 50").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_info_time_at_least(&rx, 100, Duration::from_secs(2)),
        "Lazy SMP did not keep searching beyond the ordinary hard time while pondering"
    );
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_millis(150)).is_none(),
        "pondering stopped at the ordinary hard time"
    );

    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(2)).is_some(),
        "active ponder search did not finish after ponderhit"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn disabled_ponder_option_suppresses_principal_variation_ponder_move() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 1").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "setoption name Ponder value true").unwrap();
    writeln!(stdin, "setoption name Ponder value false").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 4").unwrap();
    stdin.flush().unwrap();
    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(2))
        .expect("fixed-depth search did not return a move");
    assert!(
        !bestmove.contains(" ponder "),
        "disabled Ponder option must suppress the GUI ponder move: {bestmove}"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn enabled_ponder_option_supplies_principal_variation_ponder_move() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 1").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "setoption name Ponder value true").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(2)).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 4").unwrap();
    stdin.flush().unwrap();
    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(2))
        .expect("fixed-depth search did not return a move");
    assert!(
        bestmove.contains(" ponder "),
        "enabled Ponder option should expose the principal variation to the GUI: {bestmove}"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}
