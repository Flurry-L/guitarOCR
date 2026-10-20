use super::*;
use std::{sync::mpsc, time::Duration};

const SID: &str = "0123456789abcdef0123456789abcdef";

fn workspace() -> (tempfile::TempDir, Arc<Tasks>) {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(SID)).unwrap();
    let tasks = Tasks::new(root.path().to_owned()).unwrap();
    (root, tasks)
}

fn finished(tasks: &Tasks) -> Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let state = tasks.snapshot(SID).unwrap();
        if state["finished"].is_number() {
            return state;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "Task did not finish: {state}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn cancellation_after_the_result_is_saved_keeps_the_completed_status() {
    let (root, tasks) = workspace();
    let result = root.path().join(SID).join("result.txt");
    let output = result.clone();
    let (saved, ready) = mpsc::channel();
    let (finish, wait) = mpsc::channel();
    tasks
        .submit(SID.into(), "Test", "recognize", move |progress| {
            progress.commit(|| fs::write(output, "accepted").map_err(|e| e.to_string()))?;
            saved.send(()).unwrap();
            wait.recv_timeout(Duration::from_secs(5)).unwrap();
            Ok(())
        })
        .unwrap();
    ready.recv_timeout(Duration::from_secs(5)).unwrap();
    tasks.cancel(SID).unwrap();
    finish.send(()).unwrap();
    assert_eq!(finished(&tasks)["status"], "complete");
    assert_eq!(fs::read_to_string(result).unwrap(), "accepted");
}

#[test]
fn cancellation_before_commit_preserves_the_previous_result() {
    let (root, tasks) = workspace();
    let result = root.path().join(SID).join("result.txt");
    fs::write(&result, "previous").unwrap();
    let output = result.clone();
    let (prepared, ready) = mpsc::channel();
    let (commit, wait) = mpsc::channel();
    tasks
        .submit(SID.into(), "Test", "recognize", move |progress| {
            progress.checkpoint()?;
            prepared.send(()).unwrap();
            wait.recv_timeout(Duration::from_secs(5)).unwrap();
            progress.commit(|| fs::write(output, "replacement").map_err(|e| e.to_string()))
        })
        .unwrap();
    ready.recv_timeout(Duration::from_secs(5)).unwrap();
    tasks.cancel(SID).unwrap();
    commit.send(()).unwrap();
    let state = finished(&tasks);
    assert_eq!(state["status"], "cancelled");
    assert_eq!(state["cancellable"], false);
    assert_eq!(fs::read_to_string(result).unwrap(), "previous");
    assert_eq!(view_state(root.path())["status"], "cancelled");
}

#[test]
fn cancellation_between_stages_keeps_only_the_finished_stage() {
    let (root, tasks) = workspace();
    let first = root.path().join(SID).join("first.txt");
    let second = root.path().join(SID).join("second.txt");
    let (first_output, second_output) = (first.clone(), second.clone());
    let (saved, ready) = mpsc::channel();
    let (next, wait) = mpsc::channel();
    tasks
        .submit(SID.into(), "Test", "full", move |progress| {
            progress.commit(|| fs::write(first_output, "accepted").map_err(|e| e.to_string()))?;
            saved.send(()).unwrap();
            wait.recv_timeout(Duration::from_secs(5)).unwrap();
            progress.commit(|| fs::write(second_output, "pending").map_err(|e| e.to_string()))
        })
        .unwrap();
    ready.recv_timeout(Duration::from_secs(5)).unwrap();
    tasks.cancel(SID).unwrap();
    next.send(()).unwrap();
    assert_eq!(finished(&tasks)["status"], "cancelled");
    assert_eq!(fs::read_to_string(first).unwrap(), "accepted");
    assert!(!second.exists());
}

#[test]
fn failed_publication_finishes_the_job_and_allows_a_retry() {
    let (_root, tasks) = workspace();
    tasks
        .submit(SID.into(), "Test", "recognize", |progress| {
            progress.commit(|| panic!("Publication failed"))
        })
        .unwrap();
    let state = finished(&tasks);
    assert_eq!(state["status"], "failed");
    assert_eq!(state["cancellable"], false);
    tasks.idle(SID).unwrap();
    tasks
        .submit(SID.into(), "Retry", "recognize", |progress| {
            progress.commit(|| Ok(()))
        })
        .unwrap();
    assert_eq!(finished(&tasks)["status"], "complete");
}

fn view_state(root: &Path) -> Value {
    serde_json::from_slice(&fs::read(root.join(SID).join("job.json")).unwrap()).unwrap()
}
