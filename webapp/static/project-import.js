// Import a saved project without discarding the current editor until it succeeds.
// Navigation remains available; a late result is kept in the library instead of
// taking the user back to the editor they left.
export function createProjectImporter({
  ui, hasDrafts, confirm, setBusy, navigation, importArchive, accept, notice, refresh,
}) {
  let pending = false;
  return async (readFile, { prompt, loading, opened, background }) => {
    if (pending || ui.busy) return false;
    if ((hasDrafts() || ui.files.length) && !confirm(prompt)) return false;
    let owner = { sid: ui.sid, generation: ui.openGeneration };
    const startedAt = navigation();
    const ownsEditor = () => ui.sid === owner.sid && ui.openGeneration === owner.generation;
    pending = true;
    setBusy(true);
    notice(loading);
    try {
      const saved = await importArchive(await readFile());
      if (ownsEditor()) {
        if (navigation() === startedAt) {
          accept(saved);
          owner = { sid: ui.sid, generation: ui.openGeneration };
          notice(opened);
        } else {
          notice(background);
        }
      }
      await refresh();
      return true;
    } catch (error) {
      if (ownsEditor()) notice(error.message, true);
      return false;
    } finally {
      pending = false;
      if (ownsEditor()) setBusy(false);
    }
  };
}

export const SAMPLE_ARCHIVE = "/static/examples/Harbor-Light-synthetic-project.zip";

export async function readSample(fetchArchive = fetch) {
  let response;
  try {
    response = await fetchArchive(SAMPLE_ARCHIVE);
  } catch {
    throw new Error("暂时无法加载示例，请重试。");
  }
  if (!response.ok) throw new Error("示例文件未能加载，请重试或检查安装是否完整。");
  return new File([await response.blob()], "Harbor-Light-synthetic-project.zip", {
    type: "application/zip",
  });
}
