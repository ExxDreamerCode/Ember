use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const UCI_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

fn spawn_ember() -> (Child, Receiver<String>) {
    spawn_ember_in_dir(None)
}

fn spawn_ember_in_dir(current_dir: Option<&Path>) -> (Child, Receiver<String>) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ember"));
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(current_dir) = current_dir {
        command.current_dir(current_dir);
    }
    let mut child = command.spawn().expect("spawn Ember UCI process");
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

fn temp_book_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ember-uci-book-test-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    dir
}

fn write_startpos_book(path: &Path, raw_move: u16) {
    let startpos_polyglot_key = 0x463b_9618_1691_fc9c_u64;
    let weight = 100_u16;
    let learn = 0_u32;
    let mut data = Vec::new();
    data.extend_from_slice(&startpos_polyglot_key.to_be_bytes());
    data.extend_from_slice(&raw_move.to_be_bytes());
    data.extend_from_slice(&weight.to_be_bytes());
    data.extend_from_slice(&learn.to_be_bytes());
    fs::write(path, data).unwrap();
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

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("poll Ember UCI process") {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

fn assert_go_nodes_returns_promptly(threads: usize) {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value {threads}").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some(),
        "Ember did not finish UCI initialization"
    );

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go nodes 1").unwrap();
    stdin.flush().unwrap();

    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(2));
    if bestmove.is_none() {
        let _ = child.kill();
        let _ = child.wait();
        panic!("go nodes 1 was ignored with Threads={threads}");
    }

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("wait for Ember UCI process");
    assert!(status.success(), "Ember exited with {status}");
}

#[test]
fn go_nodes_returns_promptly_in_single_threaded_search() {
    assert_go_nodes_returns_promptly(1);
}

#[test]
fn go_nodes_returns_promptly_in_lazy_smp_search() {
    assert_go_nodes_returns_promptly(4);
}

#[test]
fn malformed_uci_input_is_rejected_without_crashing() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 1").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos moves 0000 g1f3").unwrap();
    writeln!(stdin, "position startpos moves zzzz").unwrap();
    writeln!(stdin, "position startpos moves e2e4x").unwrap();
    writeln!(stdin, "position fen 8/8/8/8/8/8/8/8 w - - 0 1").unwrap();
    writeln!(stdin, "go movetime nope depth 1").unwrap();
    stdin.flush().unwrap();

    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).is_some(),
        "Ember did not recover after malformed UCI input"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("wait for Ember UCI process");
    assert!(status.success(), "Ember exited with {status}");
}

#[test]
fn external_compact_nnue_loads_through_the_uci_option() {
    let compact_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("networks/V1/1.1.1-1.3.0/net.compact.nnue");
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");

    writeln!(stdin, "uci").unwrap();
    assert!(wait_for_line(&rx, "uciok", UCI_STARTUP_TIMEOUT).is_some());
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(
        stdin,
        "setoption name NNUE value {}",
        compact_path.display()
    )
    .unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "info string Loaded NNUE ", Duration::from_secs(5)).is_some(),
        "external compact NNUE did not report a successful load"
    );

    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(5)).is_some());
    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 1").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).is_some(),
        "search did not use the externally loaded compact NNUE"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn queued_quit_during_search_exits_cleanly() {
    let (mut child, _rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    write!(
        stdin,
        "uci\nsetoption name Hash value 16\nsetoption name Threads value 2\nsetoption name Book value\nisready\nposition startpos\ngo depth 16\nquit\n"
    )
    .unwrap();
    stdin.flush().unwrap();
    drop(stdin);

    let Some(status) = wait_for_exit(&mut child, UCI_STARTUP_TIMEOUT) else {
        let _ = child.kill();
        let _ = child.wait();
        panic!("Ember did not exit after queued quit during search");
    };
    assert!(status.success(), "Ember exited with {status}");
}

#[test]
fn input_eof_during_search_exits_cleanly() {
    let (mut child, _rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    write!(
        stdin,
        "uci\nsetoption name Hash value 16\nsetoption name Threads value 2\nsetoption name Book value\nisready\nposition startpos\ngo depth 16\n"
    )
    .unwrap();
    stdin.flush().unwrap();
    drop(stdin);

    let Some(status) = wait_for_exit(&mut child, UCI_STARTUP_TIMEOUT) else {
        let _ = child.kill();
        let _ = child.wait();
        panic!("Ember did not exit after stdin EOF during search");
    };
    assert!(status.success(), "Ember exited with {status}");
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
        wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some(),
        "Ember did not finish UCI initialization"
    );

    writeln!(stdin, "position startpos").unwrap();
    write!(stdin, "go infinite\nstop\n").unwrap();
    stdin.flush().unwrap();

    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(5));
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
fn completed_ponder_search_waits_for_ponderhit() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "option name Ponder ", UCI_STARTUP_TIMEOUT).is_some(),
        "Ember did not advertise UCI pondering"
    );
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 4").unwrap();
    writeln!(stdin, "setoption name Book value").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "readyok", Duration::from_secs(5)).is_some(),
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
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).is_some(),
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
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go ponder movetime 50").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_info_time_at_least(&rx, 100, Duration::from_secs(5)),
        "Lazy SMP did not keep searching beyond the ordinary hard time while pondering"
    );
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_millis(150)).is_none(),
        "pondering stopped at the ordinary hard time"
    );

    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).is_some(),
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
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 4").unwrap();
    stdin.flush().unwrap();
    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(5))
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
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 4").unwrap();
    stdin.flush().unwrap();
    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(5))
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

#[test]
fn enabled_ponder_option_supplies_book_ponder_move() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 1").unwrap();
    writeln!(stdin, "setoption name Ponder value true").unwrap();
    writeln!(stdin, "setoption name BookMinMoveWeight value 2000").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos moves e2e4 c7c5").unwrap();
    writeln!(stdin, "go movetime 10").unwrap();
    stdin.flush().unwrap();
    let bestmove =
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).expect("book move did not return");
    assert_eq!(
        bestmove.split_whitespace().collect::<Vec<_>>()[..],
        ["bestmove", "g1f3", "ponder", "d7d6"],
        "book move should expose a book-derived ponder reply: {bestmove}"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn random_book_move_is_opt_in_and_returns_without_searching() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        wait_for_line(&rx, "option name RandomBookMove ", UCI_STARTUP_TIMEOUT).as_deref(),
        Some("option name RandomBookMove type check default false"),
        "Ember must advertise deterministic book selection as the default"
    );

    writeln!(stdin, "setoption name RandomBookMove value true").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(5)).is_some());

    writeln!(stdin, "position startpos moves g1f3 c7c5 e2e4 a7a6").unwrap();
    writeln!(stdin, "go depth 64").unwrap();
    stdin.flush().unwrap();
    let info = wait_for_line(&rx, "info ", Duration::from_secs(5))
        .expect("random book selection did not report its result");
    assert_eq!(
        info_number(&info, "nodes"),
        Some(0),
        "random book selection unexpectedly started search: {info}"
    );
    let bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(5))
        .expect("random book selection did not return a move");
    assert!(
        ["bestmove d2d4", "bestmove c2c3"].contains(&bestmove.as_str()),
        "random book selection returned an unexpected move: {bestmove}"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}

#[test]
fn startup_ignores_local_book_until_explicitly_selected() {
    let dir = temp_book_dir();
    let book_path = dir.join("book.bin");
    // Polyglot encoding for 1.a3. This is deliberately not the embedded
    // book's normal deterministic start-position choice.
    write_startpos_book(&book_path, 0x0210);

    let (mut child, rx) = spawn_ember_in_dir(Some(&dir));
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 64").unwrap();
    stdin.flush().unwrap();
    let embedded_bestmove = wait_for_line(&rx, "bestmove ", Duration::from_secs(5))
        .expect("embedded book move did not return");
    assert_ne!(
        embedded_bestmove, "bestmove a2a3",
        "startup must ignore an unrelated working-directory book.bin"
    );

    writeln!(
        stdin,
        "setoption name Book value {}",
        book_path.to_str().unwrap()
    )
    .unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", Duration::from_secs(5)).is_some());

    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 64").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).as_deref(),
        Some("bestmove a2a3"),
        "explicit Book option should still load the selected external book"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ponder_search_bypasses_book_probe() {
    let (mut child, rx) = spawn_ember();
    let mut stdin = child.stdin.take().expect("capture Ember stdin");
    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "setoption name Hash value 16").unwrap();
    writeln!(stdin, "setoption name Threads value 1").unwrap();
    writeln!(stdin, "isready").unwrap();
    stdin.flush().unwrap();
    assert!(wait_for_line(&rx, "readyok", UCI_STARTUP_TIMEOUT).is_some());

    writeln!(stdin, "position startpos moves e2e4 c7c5").unwrap();
    writeln!(stdin, "go ponder depth 1").unwrap();
    stdin.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut searched_nodes = None;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(remaining) {
            Ok(line) if line.starts_with("info ") && info_number(&line, "depth") == Some(1) => {
                if let Some(nodes) = info_number(&line, "nodes") {
                    searched_nodes = Some(nodes);
                    if nodes > 0 {
                        break;
                    }
                }
            }
            Ok(line) if line.starts_with("bestmove ") => {
                panic!("ponder search returned before ponderhit: {line}");
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    assert!(
        searched_nodes.is_some_and(|nodes| nodes > 0),
        "go ponder in a book position must run search, got nodes={searched_nodes:?}"
    );

    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();
    assert!(
        wait_for_line(&rx, "bestmove ", Duration::from_secs(5)).is_some(),
        "ponderhit did not release the completed result"
    );

    writeln!(stdin, "quit").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    assert!(child.wait().expect("wait for Ember").success());
}
