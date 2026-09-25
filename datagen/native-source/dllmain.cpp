/**
 * gpomr_native_export.dll - Native Guitar Pro score and layout exporter.
 *
 * Strategy: load through GPCore, select one track with all its TAB staves, then
 * render the prepared engraving context and export its official score model.
 *
 * Protocol (newline-delimited JSON over \\.\pipe\gpomr_export):
 *   {"cmd":"export","input":"...","output":"...",
 *       "layout_output":"...","official_score_output":"...",
 *       "track_index":0,"tab_only":true}
 *       -> {"ok":true}
 *   {"cmd":"list_tracks","input":"..."}
 *       -> {"ok":true,"tracks":[...]}
 *   {"cmd":"release_source"} -> {"ok":true}
 *   {"cmd":"quit"} -> {"ok":true}
 */

#include <windows.h>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <new>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <limits>

// Guitar Pro ABI declarations
#include "gp_stubs.h"
#include "score_dump.h"

// Qt 头文件
#include <QCoreApplication>
#include <QTimer>
#include <QString>
#include <QFile>
#include <QDir>
#include <QThread>
#include <QFileInfo>


// ── Simple JSON helper ───────────────────────────────────────────

namespace json {

static std::string get(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        pos++;
        std::string val;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == 'n') val += '\n';
                else if (json[pos] == 't') val += '\t';
                else val += json[pos];
            } else {
                val += json[pos];
            }
            pos++;
        }
        return val;
    }
    std::string val;
    while (pos < json.size() && json[pos] != ',' && json[pos] != '}' && json[pos] != ' ') {
        val += json[pos++];
    }
    return val;
}

static std::string make(const std::initializer_list<std::pair<std::string, std::string>>& kvs) {
    std::string r = "{";
    bool first = true;
    for (auto& kv : kvs) {
        if (!first) r += ",";
        first = false;
        r += "\"" + kv.first + "\":";
        if (kv.second == "true" || kv.second == "false" || kv.second == "null" ||
            (!kv.second.empty() && (kv.second[0] == '-' || (kv.second[0] >= '0' && kv.second[0] <= '9')))) {
            r += kv.second;
        } else {
            r += "\"" + kv.second + "\"";
        }
    }
    r += "}\n";
    return r;
}

} // namespace json


// ── Logging ──────────────────────────────────────────────────────

static FILE* g_logFile = nullptr;
static int g_loggingEnabled = -1;
static char g_logPath[MAX_PATH] = {};

static void logMsg(const char* fmt, ...) {
    if (g_loggingEnabled < 0) {
        char setting[16] = {};
        const DWORD length = GetEnvironmentVariableA(
            "GPOMR_EXPORT_LOG_ENABLED", setting, sizeof(setting));
        g_loggingEnabled = length > 0
            && setting[0] != '0'
            && setting[0] != 'f'
            && setting[0] != 'F';
    }
    if (!g_loggingEnabled) {
        return;
    }
    if (!g_logFile) {
        if (!g_logPath[0]) {
            const DWORD length = GetEnvironmentVariableA(
                "GPOMR_EXPORT_LOG_PATH", g_logPath, sizeof(g_logPath));
            if (length == 0 || length >= sizeof(g_logPath)) {
                const DWORD tempLength = GetTempPathA(sizeof(g_logPath), g_logPath);
                if (tempLength == 0 || tempLength >= sizeof(g_logPath)) {
                    strcpy_s(g_logPath, "gpomr-native-export.log");
                } else {
                    strcat_s(g_logPath, "gpomr-native-export.log");
                }
            }
        }
        g_logFile = fopen(g_logPath, "a");
    }
    if (g_logFile) {
        va_list args;
        va_start(args, fmt);
        vfprintf(g_logFile, fmt, args);
        va_end(args);
        fprintf(g_logFile, "\n");
        fflush(g_logFile);
    }
}


// ── Main-thread dispatcher ───────────────────────────────────────

struct MainThreadTask {
    std::function<void()> fn;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return done; });
    }

    void complete() {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
        cv.notify_all();
    }
};

static void runOnMainThread(std::function<void()> fn) {
    if (QThread::currentThread() == QCoreApplication::instance()->thread()) {
        fn();
        return;
    }

    auto task = new MainThreadTask();
    task->fn = std::move(fn);

    QTimer::singleShot(0, QCoreApplication::instance(), [task]() {
        task->fn();
        task->complete();
    });

    task->wait();
    delete task;
}


static void* exportedFunction(HMODULE module, const char* name) {
    if (!module || !name) return nullptr;
    const auto address = GetProcAddress(module, name);
    if (!address) logMsg("[EXPORT] Missing '%s'", name);
    return reinterpret_cast<void*>(address);
}

// ── Inline Hook Infrastructure ───────────────────────────────────

constexpr size_t kAbsoluteJumpSize = 14;
constexpr size_t kMaxInlinePatchSize = 32;

struct InlineRipFixup {
    size_t displacementOffset = 0;
    size_t instructionEndOffset = 0;
};

struct InlineHook {
    void* target = nullptr;
    void* detour = nullptr;
    BYTE originalBytes[kMaxInlinePatchSize] = {};
    size_t patchSize = kAbsoluteJumpSize;
    void* trampoline = nullptr;
    bool createTrampoline = false;
    InlineRipFixup ripFixups[4] = {};
    size_t ripFixupCount = 0;
    bool installed = false;
};

static void writeAbsoluteJump(unsigned char* destination, const void* target) {
    destination[0] = 0xFF;
    destination[1] = 0x25;
    memset(destination + 2, 0, 4);
    memcpy(destination + 6, &target, sizeof(target));
}

static uintptr_t alignUp(uintptr_t value, uintptr_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static void* allocateExecutableNear(void* target, size_t bytes) {
    SYSTEM_INFO info = {};
    GetSystemInfo(&info);
    const uintptr_t granularity = info.dwAllocationGranularity;
    const uintptr_t targetAddress = reinterpret_cast<uintptr_t>(target);
    const uintptr_t processMin = reinterpret_cast<uintptr_t>(info.lpMinimumApplicationAddress);
    const uintptr_t processMax = reinterpret_cast<uintptr_t>(info.lpMaximumApplicationAddress);
    const uintptr_t relativeLimit = 0x7FFF0000ull;
    const uintptr_t minAddress = targetAddress > relativeLimit
        ? (std::max)(processMin, targetAddress - relativeLimit)
        : processMin;
    const uintptr_t maxAddress = targetAddress < processMax - relativeLimit
        ? targetAddress + relativeLimit
        : processMax;

    auto tryRegion = [&](const MEMORY_BASIC_INFORMATION& region) -> void* {
        if (region.State != MEM_FREE) return nullptr;
        uintptr_t begin = (std::max)(
            reinterpret_cast<uintptr_t>(region.BaseAddress),
            minAddress);
        uintptr_t end = (std::min)(
            reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize,
            maxAddress);
        uintptr_t candidate = alignUp(begin, granularity);
        if (candidate > end || bytes > end - candidate) return nullptr;
        return VirtualAlloc(
            reinterpret_cast<void*>(candidate),
            bytes,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_EXECUTE_READWRITE);
    };

    uintptr_t cursor = targetAddress;
    while (cursor >= minAddress) {
        MEMORY_BASIC_INFORMATION region = {};
        if (!VirtualQuery(reinterpret_cast<void*>(cursor), &region, sizeof(region))) break;
        if (void* allocated = tryRegion(region)) return allocated;
        uintptr_t regionBegin = reinterpret_cast<uintptr_t>(region.BaseAddress);
        if (regionBegin <= minAddress) break;
        cursor = regionBegin - 1;
    }

    cursor = targetAddress;
    while (cursor <= maxAddress) {
        MEMORY_BASIC_INFORMATION region = {};
        if (!VirtualQuery(reinterpret_cast<void*>(cursor), &region, sizeof(region))) break;
        if (void* allocated = tryRegion(region)) return allocated;
        uintptr_t next = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
        if (next <= cursor || next > maxAddress) break;
        cursor = next;
    }
    return nullptr;
}

static bool buildInlineTrampoline(InlineHook* hook) {
    if (!hook->createTrampoline) return true;
    if (hook->trampoline) return true;

    const size_t trampolineSize = hook->patchSize + kAbsoluteJumpSize;
    auto* trampoline = static_cast<unsigned char*>(
        allocateExecutableNear(hook->target, trampolineSize));
    if (!trampoline) {
        logMsg("[INLINE] Failed to allocate near trampoline for %p", hook->target);
        return false;
    }
    memcpy(trampoline, hook->originalBytes, hook->patchSize);

    for (size_t i = 0; i < hook->ripFixupCount; ++i) {
        const auto& fixup = hook->ripFixups[i];
        if (fixup.displacementOffset + sizeof(int32_t) > hook->patchSize
                || fixup.instructionEndOffset > hook->patchSize) {
            VirtualFree(trampoline, 0, MEM_RELEASE);
            return false;
        }
        int32_t originalDisplacement = 0;
        memcpy(
            &originalDisplacement,
            hook->originalBytes + fixup.displacementOffset,
            sizeof(originalDisplacement));
        const intptr_t originalTarget =
            reinterpret_cast<intptr_t>(hook->target)
            + static_cast<intptr_t>(fixup.instructionEndOffset)
            + originalDisplacement;
        const intptr_t relocatedEnd =
            reinterpret_cast<intptr_t>(trampoline)
            + static_cast<intptr_t>(fixup.instructionEndOffset);
        const intptr_t relocatedDisplacement = originalTarget - relocatedEnd;
        if (relocatedDisplacement < (std::numeric_limits<int32_t>::min)()
                || relocatedDisplacement > (std::numeric_limits<int32_t>::max)()) {
            logMsg("[INLINE] RIP relocation is out of range for %p", hook->target);
            VirtualFree(trampoline, 0, MEM_RELEASE);
            return false;
        }
        const int32_t encodedDisplacement = static_cast<int32_t>(relocatedDisplacement);
        memcpy(
            trampoline + fixup.displacementOffset,
            &encodedDisplacement,
            sizeof(encodedDisplacement));
    }

    writeAbsoluteJump(
        trampoline + hook->patchSize,
        static_cast<unsigned char*>(hook->target) + hook->patchSize);
    FlushInstructionCache(GetCurrentProcess(), trampoline, trampolineSize);
    DWORD oldProtect = 0;
    VirtualProtect(trampoline, trampolineSize, PAGE_EXECUTE_READ, &oldProtect);
    hook->trampoline = trampoline;
    logMsg("[INLINE] Trampoline target=%p trampoline=%p patch=%zu",
           hook->target,
           hook->trampoline,
           hook->patchSize);
    return true;
}

static bool installInlineHook(InlineHook* hook) {
    if (!hook->target || !hook->detour
            || hook->patchSize < kAbsoluteJumpSize
            || hook->patchSize > kMaxInlinePatchSize) {
        return false;
    }
    if (hook->installed) return true;

    DWORD oldProt;
    if (!VirtualProtect(
            hook->target,
            hook->patchSize,
            PAGE_EXECUTE_READWRITE,
            &oldProt)) {
        logMsg("[INLINE] VirtualProtect failed: %lu", GetLastError());
        return false;
    }

    memcpy(hook->originalBytes, hook->target, hook->patchSize);
    if (!buildInlineTrampoline(hook)) {
        DWORD ignored = 0;
        VirtualProtect(hook->target, hook->patchSize, oldProt, &ignored);
        return false;
    }

    memset(hook->target, 0x90, hook->patchSize);
    writeAbsoluteJump(static_cast<unsigned char*>(hook->target), hook->detour);
    DWORD ignored = 0;
    VirtualProtect(hook->target, hook->patchSize, oldProt, &ignored);
    FlushInstructionCache(GetCurrentProcess(), hook->target, hook->patchSize);

    hook->installed = true;
    return true;
}


// ── Hook State & Functions ───────────────────────────────────────

static void removeInlineHook(InlineHook* hook) {
    if (!hook->installed) return;
    DWORD oldProt;
    VirtualProtect(hook->target, hook->patchSize, PAGE_EXECUTE_READWRITE, &oldProt);
    memcpy(hook->target, hook->originalBytes, hook->patchSize);
    VirtualProtect(hook->target, hook->patchSize, oldProt, &oldProt);
    FlushInstructionCache(GetCurrentProcess(), hook->target, hook->patchSize);
    hook->installed = false;
}

// ── Hook Installation ────────────────────────────────────────────

static void* guitarProMainVa(uintptr_t originalVa);

static InlineHook g_pageDecorationHook = {};
static thread_local bool g_forcePageDecorations = false;

using fn_PageDecorationUpdate = void (__fastcall *)(void*);

static void __fastcall hookedPageDecorationUpdate(void* state) {
    if (g_forcePageDecorations && state) {
        auto* dirtyMask = reinterpret_cast<unsigned int*>(
            static_cast<unsigned char*>(state) + 168);
        *dirtyMask |= 0x1Fu;
    }
    auto original = reinterpret_cast<fn_PageDecorationUpdate>(
        g_pageDecorationHook.trampoline);
    original(state);
}

static bool installPageDecorationHook() {
    if (g_pageDecorationHook.installed) return true;
    g_pageDecorationHook.target = guitarProMainVa(0x7FF68F2F88D0);
    g_pageDecorationHook.detour = reinterpret_cast<void*>(
        hookedPageDecorationUpdate);
    g_pageDecorationHook.patchSize = 15;
    g_pageDecorationHook.createTrampoline = true;
    if (!installInlineHook(&g_pageDecorationHook)) {
        logMsg("[CORE-DIRECT] Failed to install page decoration hook");
        return false;
    }
    logMsg("[CORE-DIRECT] Page decoration hook installed @ %p",
           g_pageDecorationHook.target);
    return true;
}

class ScopedPageDecorationForce {
public:
    ScopedPageDecorationForce()
        : previous_(g_forcePageDecorations) {
        g_forcePageDecorations = true;
    }

    ~ScopedPageDecorationForce() {
        g_forcePageDecorations = previous_;
    }

private:
    bool previous_;
};

// ── Converter ────────────────────────────────────────────────────

struct ConvertResult {
    bool ok = false;
    std::string errorCode;
    std::string error;
};

struct SourceTrackInfo {
    int sourceTrackIndex = -1;
    std::string name;
    std::string instrumentKind;
};

struct TrackListResult {
    bool ok = false;
    std::string errorCode;
    std::string error;
    std::vector<SourceTrackInfo> tracks;
};

static void* guitarProMainVa(uintptr_t originalVa) {
    constexpr uintptr_t kOriginalImageBase = 0x7FF68E930000;
    if (originalVa < kOriginalImageBase) return nullptr;
    HMODULE exe = GetModuleHandleA("GuitarPro.exe");
    if (!exe) return nullptr;
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(exe) + (originalVa - kOriginalImageBase));
}

static std::unique_ptr<am::filesystem::RealFileSystem> g_directFileSystem;
static std::shared_ptr<gp::core::Track> g_directWarmTrack;
static std::shared_ptr<gp::core::Score> g_sessionScore;
static std::string g_sessionScorePath;
static std::once_flag g_directFileSystemOnce;
static std::atomic<bool> g_directReady{false};
static std::atomic<bool> g_directReadyProbeStarted{false};
static HANDLE g_directReadyEvent = NULL;
static std::mutex g_directConvertMutex;
static int g_directProbeCount = 0;

static void ensureDirectFileSystem() {
    std::call_once(g_directFileSystemOnce, []() {
        QString rootPath = QCoreApplication::applicationDirPath();
        if (!rootPath.endsWith('/')) rootPath += '/';
        g_directFileSystem = std::make_unique<am::filesystem::RealFileSystem>(rootPath);
    });
}

static bool probeDirectReadyOnMainThread() {
    if (g_directReady.load()) return true;

    auto* app = QCoreApplication::instance();
    if (!app) return false;

    ensureDirectFileSystem();
    if (!g_directFileSystem) return false;

    try {
        gp::core::Core::instance().waitLazyConf();
        gp::core::Core::instance().loadNotationPatches();
        g_directWarmTrack = gp::core::Core::instance().findGMInstrument(30);
        if (g_directWarmTrack) {
            g_directReady.store(true);
            if (g_directReadyEvent) {
                SetEvent(g_directReadyEvent);
            }
            logMsg("[DIRECT] Core ready signal emitted");
            return true;
        }
        if ((++g_directProbeCount % 20) == 1) {
            logMsg("[DIRECT] Core probe: GM track not ready yet");
        }
    } catch (...) {
        if ((++g_directProbeCount % 20) == 1) {
            logMsg("[DIRECT] Core probe threw; will retry");
        }
    }
    return false;
}

static void startDirectReadyProbeOnMainThread() {
    auto* app = QCoreApplication::instance();
    if (!app || g_directReady.load()) return;

    auto* timer = new QTimer(app);
    timer->setInterval(250);
    QObject::connect(timer, &QTimer::timeout, [timer]() {
        if (probeDirectReadyOnMainThread()) {
            timer->stop();
            timer->deleteLater();
        }
    });
    timer->start();
    QTimer::singleShot(0, app, []() {
        probeDirectReadyOnMainThread();
    });
    logMsg("[DIRECT] Core ready probe timer started");
}

static DWORD WINAPI directReadyProbeThread(LPVOID) {
    while (!QCoreApplication::instance()) {
        Sleep(50);
    }
    QTimer::singleShot(0, QCoreApplication::instance(), []() {
        startDirectReadyProbeOnMainThread();
    });
    return 0;
}

static void startDirectReadyProbe() {
    if (g_directReadyProbeStarted.exchange(true)) return;
    HANDLE thread = CreateThread(NULL, 0, directReadyProbeThread, NULL, 0, NULL);
    if (thread) {
        CloseHandle(thread);
    } else {
        logMsg("[DIRECT] Failed to start ready probe thread: %lu", GetLastError());
    }
}

static bool waitForDirectReadyOnMainThread(int timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (probeDirectReadyOnMainThread()) return true;
        QCoreApplication::processEvents();
        Sleep(50);
    }
    return g_directReady.load();
}

struct NativeMasterBarRange {
    int firstBarIndex = -1;
    int lastBarIndex = -1;
    bool multirest = false;
};

struct NativeEventLocation {
    int staffIndex = -1;
    int measureIndex = -1;
    int voiceIndex = -1;
    int eventIndex = -1;
};

struct NativeTempoOwner {
    int masterMeasureIndex = -1;
    double position = 0.0;
    bool initial = false;
};

struct DirectExportState {
    std::string outputPath;
    std::string layoutPath;
    std::vector<NativeMasterBarRange> masterBarRanges;
    const gp::core::style::Stylesheet* stylesheet = nullptr;
    std::map<const gp::core::Beat*, NativeEventLocation> beatLocations;
    std::vector<NativeTempoOwner> tempoOwners;
    std::vector<std::string> noteGeometryErrors;
    int sourceTrackIndex = -1;
    bool restViewsResolved = true;
    bool lyricEventsResolved = true;
    bool tempoIndicationsResolved = true;
    bool captureNoteGeometry = true;
    int dpi = 300;
    bool pdfOk = false;
    bool layoutOk = false;
    bool scoreViewMultiRest = false;
    std::string layoutError;
};

static thread_local DirectExportState* g_directExportState = nullptr;

struct NativeLayoutPoint {
    double x = 0.0;
    double y = 0.0;
};

struct NativeLayoutRect {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct NativeLayoutPage {
    int index = 0;
    NativeLayoutRect rect;
};

struct NativeTempoViewCapture {
    const void* identity = nullptr;
    int firstSystemMeasureIndex = -1;
    NativeLayoutRect abs;
};

struct NativeTempoIndication {
    int page = -1;
    int systemIndex = -1;
    int masterMeasureIndex = -1;
    double position = 0.0;
    NativeLayoutRect pageLocal;
    bool initial = false;
};

struct NativePainterTranslation {
    double x = 0.0;
    double y = 0.0;
    bool translationOnly = true;
};

struct NativeTimeSignature {
    int staffIndex = 0;
    int measureIndex = -1;
    unsigned int numerator = 0;
    unsigned int denominator = 0;
};

struct NativeKeySignature {
    int staffIndex = 0;
    int measureIndex = -1;
    int accidentalCount = 0;
};

struct NativeRestView {
    NativeEventLocation location;
    int visibilityCode = 0;
};

struct NativeLayoutMeasureView {
    int staffIndex = -1;
    int measureIndex = -1;
    NativeLayoutRect abs;
};

struct NativeNoteGlyph {
    NativeEventLocation location;
    int noteIndex = -1;
    int stringIndex = -1;
    int visibilityCode = 0;
    int page = -1;
    int systemIndex = -1;
    std::string text;
    NativeLayoutPoint origin;
    NativeLayoutRect bounds;
    NativeLayoutRect layoutBounds;
    NativeLayoutRect pageBounds;
    NativeLayoutRect pageLayoutBounds;
    NativeLayoutPoint pageOrigin;
};

struct NativeLayoutMeasure {
    int measureIndex = -1;
    NativeLayoutRect pageLocal;
};

struct NativeLayoutSystem {
    int page = -1;
    int systemIndex = -1;
    int firstMeasureIndex = -1;
    int lastMeasureIndex = -1;
    NativeLayoutRect abs;
    NativeLayoutRect pageLocal;
    std::string trackLabel;
    std::vector<NativeLayoutMeasure> measures;
    std::vector<NativeTimeSignature> timeSignatures;
    std::vector<NativeKeySignature> keySignatures;
};

static InlineHook g_layoutSystemViewHook = {};
static InlineHook g_layoutBarViewHook = {};
static InlineHook g_layoutRestViewHook = {};
static InlineHook g_layoutFretViewHook = {};
static InlineHook g_layoutBandElementViewHook = {};
static InlineHook g_layoutTempoIndicationRenderHook = {};
static InlineHook g_layoutPaintDevicePushMatrixHook = {};
static InlineHook g_layoutPaintDevicePopMatrixHook = {};
using fn_ViewBoundingRect = const am::painting::Rect& (__fastcall *)(const void*);
using fn_BarViewElement = const std::shared_ptr<am::painting::IElement>&
    (__fastcall *)(const void*);
using fn_NoteViewBeatModel = std::shared_ptr<gp::core::Beat>* (__fastcall *)(
    const void*, std::shared_ptr<gp::core::Beat>*);
using fn_NoteViewVoiceIndex = int (__fastcall *)(const void*);
using fn_NoteViewNoteModel = std::shared_ptr<gp::core::Note>* (__fastcall *)(
    const void*, std::shared_ptr<gp::core::Note>*);
using fn_NoteParentBeat = gp::core::Beat* (__fastcall *)(const gp::core::Note*);
using fn_FretViewText = const std::string& (__fastcall *)(const void*);
using fn_FretViewFont = const void* (__fastcall *)(const void*);
using fn_StyleTextBounds = am::painting::Rect* (__fastcall *)(
    const void*, am::painting::Rect*, const std::string*, const void*, int, int, int);
using fn_ViewVisibility = gp::core::view::Visibility (__fastcall *)(const void*);
using fn_BandElementViewType = gp::core::view::BandElementType (__fastcall *)(
    const void*);
using fn_LyricBandElementAttachedBeat =
    const std::shared_ptr<const gp::core::Beat>& (__fastcall *)(const void*);
using fn_TempoIndicationRender = void (__fastcall *)(
    void*, void*, const void*, const void*);
using fn_PaintDevicePushMatrix = void (__fastcall *)(void*, const void*);
using fn_MatrixValue = double (__fastcall *)(const void*);
using fn_MatrixIsTranslation = bool (__fastcall *)(const void*);
static fn_ViewBoundingRect g_viewBoundingRect = nullptr;
static fn_BarViewElement g_barViewTimeSignatureElement = nullptr;
static fn_BarViewElement g_barViewKeySignatureElement = nullptr;
static fn_NoteViewBeatModel g_noteViewBeatModel = nullptr;
static fn_NoteViewVoiceIndex g_noteViewVoiceIndex = nullptr;
static fn_NoteViewNoteModel g_noteViewNoteModel = nullptr;
static fn_NoteParentBeat g_noteParentBeat = nullptr;
static fn_FretViewText g_fretViewText = nullptr;
static fn_FretViewFont g_fretViewFont = nullptr;
static fn_StyleTextBounds g_styleTextBounds = nullptr;
static fn_ViewVisibility g_viewVisibility = nullptr;
static fn_BandElementViewType g_bandElementViewType = nullptr;
static fn_LyricBandElementAttachedBeat g_lyricBandElementAttachedBeat = nullptr;
static fn_TempoIndicationRender g_offscreenDrawingElementRender = nullptr;
static fn_MatrixValue g_matrixTx = nullptr;
static fn_MatrixValue g_matrixTy = nullptr;
static fn_MatrixIsTranslation g_matrixIsTranslation = nullptr;
static std::once_flag g_layoutHooksOnce;
static std::atomic<bool> g_layoutHooksReady{false};
static std::atomic<bool> g_restViewHookReady{false};
static std::atomic<bool> g_noteGeometryHookReady{false};
static std::atomic<bool> g_lyricBandElementHookReady{false};
static std::atomic<bool> g_tempoIndicationHooksReady{false};
static thread_local std::vector<NativeLayoutPage> g_layoutPages;
static thread_local std::vector<NativeLayoutSystem> g_layoutSystems;
static thread_local std::vector<NativeTimeSignature> g_timeSignatures;
static thread_local std::vector<NativeKeySignature> g_keySignatures;
static thread_local std::vector<NativeLayoutMeasureView> g_layoutMeasureViews;
static thread_local std::vector<NativeRestView> g_restViews;
static thread_local std::vector<NativeNoteGlyph> g_noteGlyphs;
static thread_local std::vector<NativeEventLocation> g_lyricEvents;
static thread_local std::vector<NativeTempoViewCapture> g_tempoViewCaptures;
static thread_local std::vector<NativeTempoIndication> g_tempoIndications;
static thread_local std::map<void*, std::vector<NativePainterTranslation>>
    g_paintDeviceTranslations;
static thread_local int g_renderedSystemFirstMeasureIndex = -1;
static thread_local int g_tempoBandElementRenderDepth = 0;
static thread_local bool g_layoutDumpActive = false;

static bool isFiniteRect(const NativeLayoutRect& r) {
    return std::isfinite(r.x) && std::isfinite(r.y) &&
           std::isfinite(r.w) && std::isfinite(r.h) &&
           r.w > 0.0 && r.h > 0.0;
}

static bool callViewBoundingRect(void* view, NativeLayoutRect* out);

static NativeLayoutRect rectFromView(void* view) {
    NativeLayoutRect result = {};
    callViewBoundingRect(view, &result);
    return result;
}

static NativeLayoutRect translateRect(const NativeLayoutRect& r, const NativeLayoutPoint& p) {
    return {r.x + p.x, r.y + p.y, r.w, r.h};
}

static NativeLayoutRect unionRect(
        const NativeLayoutRect& left,
        const NativeLayoutRect& right) {
    const double x = (std::min)(left.x, right.x);
    const double y = (std::min)(left.y, right.y);
    const double x2 = (std::max)(left.x + left.w, right.x + right.w);
    const double y2 = (std::max)(left.y + left.h, right.y + right.h);
    return {x, y, x2 - x, y2 - y};
}

static void markTempoIndicationsUnresolved(const char* reason) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->tempoIndicationsResolved) return;
    state->tempoIndicationsResolved = false;
    g_tempoViewCaptures.clear();
    g_tempoIndications.clear();
    logMsg("[LAYOUT] TempoIndication gate unresolved: %s", reason);
}

static void* paintDeviceFromPainter(void* painter) {
    if (!painter) return nullptr;
    __try {
        void* painterState = *reinterpret_cast<void**>(painter);
        return painterState ? *reinterpret_cast<void**>(painterState) : nullptr;
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
}

static NativePainterTranslation currentPainterTranslation(void* painter) {
    void* device = paintDeviceFromPainter(painter);
    const auto found = g_paintDeviceTranslations.find(device);
    if (found == g_paintDeviceTranslations.end() || found->second.empty()) {
        return {};
    }
    return found->second.back();
}

static void __fastcall hookedPaintDevicePushMatrix(
        void* device,
        const void* matrix) {
    auto original = reinterpret_cast<fn_PaintDevicePushMatrix>(
        g_layoutPaintDevicePushMatrixHook.trampoline);
    if (g_layoutDumpActive && device && matrix) {
        auto& stack = g_paintDeviceTranslations[device];
        const NativePainterTranslation parent = stack.empty()
            ? NativePainterTranslation{}
            : stack.back();
        NativePainterTranslation current = parent;
        if (!g_matrixTx || !g_matrixTy || !g_matrixIsTranslation) {
            current.translationOnly = false;
        } else {
            __try {
                current.x += g_matrixTx(matrix);
                current.y += g_matrixTy(matrix);
                current.translationOnly = current.translationOnly
                    && g_matrixIsTranslation(matrix);
            } __except(EXCEPTION_EXECUTE_HANDLER) {
                current.translationOnly = false;
            }
        }
        stack.push_back(current);
    }
    original(device, matrix);
}

static void __fastcall hookedPaintDevicePopMatrix(void* device) {
    if (g_layoutDumpActive && device) {
        const auto found = g_paintDeviceTranslations.find(device);
        if (found != g_paintDeviceTranslations.end() && !found->second.empty()) {
            found->second.pop_back();
        }
    }
    __try {
        auto** vtable = *reinterpret_cast<void***>(device);
        auto original = reinterpret_cast<void (__fastcall *)(void*)>(vtable[10]);
        original(device);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        markTempoIndicationsUnresolved("IPaintDevice matrix pop failed");
    }
}

static void captureInitialTempoIndication(void* self, void* painter) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->tempoIndicationsResolved || !self || !painter
            || g_tempoBandElementRenderDepth != 0) return;
    const auto owner = std::find_if(
        state->tempoOwners.begin(),
        state->tempoOwners.end(),
        [](const NativeTempoOwner& candidate) { return candidate.initial; });
    if (owner == state->tempoOwners.end()) {
        markTempoIndicationsUnresolved(
            "rendered page TempoIndication has no initial tempo owner");
        return;
    }
    if (std::any_of(
            g_tempoViewCaptures.begin(),
            g_tempoViewCaptures.end(),
            [&](const NativeTempoViewCapture& capture) {
                return capture.identity == self
                    && capture.firstSystemMeasureIndex < 0;
            })) return;
    try {
        const auto bbox = static_cast<const am::painting::IElement*>(self)->BBox();
        NativeLayoutRect abs = {bbox.x, bbox.y, bbox.w, bbox.h};
        const auto translation = currentPainterTranslation(painter);
        if (!translation.translationOnly) {
            markTempoIndicationsUnresolved(
                "page TempoIndication is under a non-translation painter matrix");
            return;
        }
        abs = translateRect(abs, {translation.x, translation.y});
        if (!isFiniteRect(abs)) {
            markTempoIndicationsUnresolved(
                "page TempoIndication has an invalid official bounding box");
            return;
        }
        g_tempoViewCaptures.push_back({self, -1, abs});
    } catch (...) {
        markTempoIndicationsUnresolved(
            "official page TempoIndication bounding box getter failed");
    }
}

static void __fastcall hookedTempoIndicationRender(
        void* self,
        void* painter,
        const void* pen,
        const void* brush) {
    if (g_layoutDumpActive) captureInitialTempoIndication(self, painter);
    g_offscreenDrawingElementRender(self, painter, pen, brush);
}

static void setLayoutError(const std::string& message) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->layoutError.empty()) return;
    state->layoutError = message;
    logMsg("[LAYOUT] %s", state->layoutError.c_str());
}

static int uniqueContainingPageForRect(
        const NativeLayoutRect& abs,
        NativeLayoutRect* pageLocal,
        bool required) {
    constexpr double kContainmentTolerance = 1e-6;
    int containingPage = -1;
    int containingCount = 0;
    NativeLayoutRect containingPageRect = {};
    for (const auto& page : g_layoutPages) {
        const bool anchorContained =
            abs.x >= page.rect.x - kContainmentTolerance
            && abs.y >= page.rect.y - kContainmentTolerance
            && abs.x < page.rect.x + page.rect.w + kContainmentTolerance
            && abs.y < page.rect.y + page.rect.h + kContainmentTolerance;
        if (!anchorContained) continue;
        containingPage = page.index;
        containingPageRect = page.rect;
        ++containingCount;
    }
    if (pageLocal) {
        if (containingCount == 1) {
            const double left = (std::max)(abs.x, containingPageRect.x);
            const double top = (std::max)(abs.y, containingPageRect.y);
            const double right = (std::min)(
                abs.x + abs.w,
                containingPageRect.x + containingPageRect.w);
            const double bottom = (std::min)(
                abs.y + abs.h,
                containingPageRect.y + containingPageRect.h);
            *pageLocal = {
                left - containingPageRect.x,
                top - containingPageRect.y,
                right - left,
                bottom - top,
            };
        } else {
            *pageLocal = abs;
        }
    }
    if (containingCount != 1) {
        logMsg(
            "[LAYOUT] Uncontained bbox x=%.6f y=%.6f w=%.6f h=%.6f pages=%zu count=%d",
            abs.x,
            abs.y,
            abs.w,
            abs.h,
            g_layoutPages.size(),
            containingCount);
        if (required) {
            setLayoutError(
                "rendered layout anchor must belong to exactly one page; count="
                + std::to_string(containingCount));
        }
        return -1;
    }
    if (pageLocal && !isFiniteRect(*pageLocal)) {
        if (required) {
            setLayoutError("rendered layout bbox has no visible page intersection");
        }
        return -1;
    }
    return containingPage;
}

static std::string jsonEscape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

static bool addLayoutSystem(NativeLayoutSystem system) {
    if (!g_layoutDumpActive
            || (g_directExportState
                && !g_directExportState->layoutError.empty())) return false;
    if (!isFiniteRect(system.abs)) {
        setLayoutError("SystemView has an invalid bounding rectangle");
        return false;
    }
    system.page = uniqueContainingPageForRect(
        system.abs, &system.pageLocal, true);
    if (system.page < 1) return false;
    g_layoutSystems.push_back(std::move(system));
    return true;
}

static void collectLayoutPages(void* context) {
    g_layoutPages.clear();
    if (!context) return;
    auto* words = reinterpret_cast<uintptr_t*>(context);
    uintptr_t begin = words[39];
    uintptr_t end = words[40];
    int count = static_cast<int>((end - begin) / 208);
    for (int i = 0; i < count; ++i) {
        auto* pageView = reinterpret_cast<void*>(begin + static_cast<uintptr_t>(i) * 208);
        NativeLayoutRect rect = rectFromView(pageView);
        if (isFiniteRect(rect)) {
            g_layoutPages.push_back({i + 1, rect});
        }
    }
}

static NativeLayoutSystem* layoutSystemForMeasure(int measureIndex) {
    for (auto& system : g_layoutSystems) {
        if (measureIndex >= system.firstMeasureIndex
                && measureIndex < system.lastMeasureIndex) {
            return &system;
        }
    }
    return nullptr;
}

static bool finalizeTempoIndications() {
    DirectExportState* state = g_directExportState;
    if (!state || !state->tempoIndicationsResolved) {
        setLayoutError("official TempoIndication capture is unresolved");
        return false;
    }
    g_tempoIndications.clear();

    std::vector<const NativeTempoOwner*> initialOwners;
    std::vector<const NativeTempoViewCapture*> initialCaptures;
    for (const auto& owner : state->tempoOwners) {
        if (owner.initial) initialOwners.push_back(&owner);
    }
    for (const auto& capture : g_tempoViewCaptures) {
        if (capture.firstSystemMeasureIndex < 0) {
            initialCaptures.push_back(&capture);
        }
    }
    if (initialOwners.size() != initialCaptures.size()
            || initialOwners.size() > 1) {
        setLayoutError(
            "initial TempoIndication owner/view count differs: owners="
            + std::to_string(initialOwners.size())
            + " views=" + std::to_string(initialCaptures.size()));
        return false;
    }
    if (!initialOwners.empty()) {
        NativeLayoutRect pageLocal = {};
        const int page = uniqueContainingPageForRect(
            initialCaptures.front()->abs, &pageLocal, true);
        if (page != 1 || !isFiniteRect(pageLocal)) {
            setLayoutError("initial TempoIndication is not on the first page");
            return false;
        }
        g_tempoIndications.push_back({
            page,
            -1,
            initialOwners.front()->masterMeasureIndex,
            initialOwners.front()->position,
            pageLocal,
            true,
        });
    }

    size_t pairedAutomationViews = 0;
    for (const auto& system : g_layoutSystems) {
        std::vector<const NativeTempoOwner*> owners;
        std::vector<const NativeTempoViewCapture*> captures;
        for (const auto& owner : state->tempoOwners) {
            if (!owner.initial
                    && owner.masterMeasureIndex >= system.firstMeasureIndex
                    && owner.masterMeasureIndex < system.lastMeasureIndex) {
                owners.push_back(&owner);
            }
        }
        for (const auto& capture : g_tempoViewCaptures) {
            if (capture.firstSystemMeasureIndex == system.firstMeasureIndex) {
                captures.push_back(&capture);
            }
        }
        std::sort(
            owners.begin(),
            owners.end(),
            [](const NativeTempoOwner* left, const NativeTempoOwner* right) {
                return std::tie(left->masterMeasureIndex, left->position)
                    < std::tie(right->masterMeasureIndex, right->position);
            });
        std::sort(
            captures.begin(),
            captures.end(),
            [](const NativeTempoViewCapture* left,
               const NativeTempoViewCapture* right) {
                return std::tie(left->abs.x, left->abs.y)
                    < std::tie(right->abs.x, right->abs.y);
            });
        if (owners.size() != captures.size()) {
            setLayoutError(
                "tempo automation owner/view count differs in system="
                + std::to_string(system.systemIndex)
                + ": owners=" + std::to_string(owners.size())
                + " views=" + std::to_string(captures.size()));
            return false;
        }
        for (size_t index = 1; index < owners.size(); ++index) {
            if (owners[index - 1]->masterMeasureIndex
                        == owners[index]->masterMeasureIndex
                    && owners[index - 1]->position == owners[index]->position) {
                setLayoutError(
                    "tempo automation owners are not unique at measure="
                    + std::to_string(owners[index]->masterMeasureIndex));
                return false;
            }
        }
        for (size_t index = 0; index < owners.size(); ++index) {
            NativeLayoutRect pageLocal = {};
            const int page = uniqueContainingPageForRect(
                captures[index]->abs, &pageLocal, true);
            if (page != system.page || !isFiniteRect(pageLocal)) {
                setLayoutError(
                    "tempo automation view is outside its SystemView page: system="
                    + std::to_string(system.systemIndex));
                return false;
            }
            g_tempoIndications.push_back({
                page,
                system.systemIndex,
                owners[index]->masterMeasureIndex,
                owners[index]->position,
                pageLocal,
                false,
            });
            ++pairedAutomationViews;
        }
    }

    const size_t automationOwnerCount = static_cast<size_t>(std::count_if(
        state->tempoOwners.begin(),
        state->tempoOwners.end(),
        [](const NativeTempoOwner& owner) { return !owner.initial; }));
    if (pairedAutomationViews != automationOwnerCount
            || g_tempoViewCaptures.size()
                != pairedAutomationViews + initialCaptures.size()) {
        setLayoutError("not every official TempoIndication has a unique owner");
        return false;
    }
    std::sort(
        g_tempoIndications.begin(),
        g_tempoIndications.end(),
        [](const NativeTempoIndication& left,
           const NativeTempoIndication& right) {
            return std::tie(
                left.masterMeasureIndex,
                left.position,
                left.initial)
                < std::tie(
                    right.masterMeasureIndex,
                    right.position,
                    right.initial);
        });
    return true;
}

static bool finalizeNativeLayoutSystems() {
    if (g_layoutPages.empty() || g_layoutSystems.empty()) {
        setLayoutError(
            "Page/SystemView skeleton is empty: pages="
            + std::to_string(g_layoutPages.size())
            + " systems=" + std::to_string(g_layoutSystems.size()));
        return false;
    }
    for (size_t index = 0; index < g_layoutPages.size(); ++index) {
        if (g_layoutPages[index].index != static_cast<int>(index + 1)
                || !isFiniteRect(g_layoutPages[index].rect)) {
            setLayoutError(
                "invalid page table at position " + std::to_string(index)
                + ": page_index=" + std::to_string(g_layoutPages[index].index));
            return false;
        }
    }

    std::sort(
        g_layoutSystems.begin(),
        g_layoutSystems.end(),
        [](const NativeLayoutSystem& left, const NativeLayoutSystem& right) {
            return left.firstMeasureIndex < right.firstMeasureIndex;
        });
    int nextMeasureIndex = 0;
    int previousPage = 1;
    for (size_t index = 0; index < g_layoutSystems.size(); ++index) {
        auto& system = g_layoutSystems[index];
        if (system.firstMeasureIndex != nextMeasureIndex
                || system.lastMeasureIndex <= system.firstMeasureIndex
                || system.page < previousPage
                || system.page > static_cast<int>(g_layoutPages.size())) {
            setLayoutError(
                "invalid SystemView at position " + std::to_string(index)
                + ": first=" + std::to_string(system.firstMeasureIndex)
                + " last=" + std::to_string(system.lastMeasureIndex)
                + " expected_first=" + std::to_string(nextMeasureIndex)
                + " page=" + std::to_string(system.page)
                + " previous_page=" + std::to_string(previousPage)
                + " page_count=" + std::to_string(g_layoutPages.size()));
            return false;
        }
        system.systemIndex = static_cast<int>(index);
        nextMeasureIndex = system.lastMeasureIndex;
        previousPage = system.page;
    }
    DirectExportState* state = g_directExportState;
    if (!state || state->masterBarRanges.empty()
            || nextMeasureIndex != state->masterBarRanges.back().lastBarIndex) {
        setLayoutError(
            "SystemView ranges do not cover the selected track: covered="
            + std::to_string(nextMeasureIndex)
            + " expected="
            + std::to_string(
                !state || state->masterBarRanges.empty()
                    ? -1
                    : state->masterBarRanges.back().lastBarIndex));
        return false;
    }

    for (const auto& signature : g_timeSignatures) {
        if (auto* system = layoutSystemForMeasure(signature.measureIndex)) {
            system->timeSignatures.push_back(signature);
        }
    }
    for (const auto& signature : g_keySignatures) {
        if (auto* system = layoutSystemForMeasure(signature.measureIndex)) {
            system->keySignatures.push_back(signature);
        }
    }
    for (const auto& view : g_layoutMeasureViews) {
        auto* system = layoutSystemForMeasure(view.measureIndex);
        if (!system) {
            setLayoutError(
                "BarView measure has no owning SystemView: measure="
                + std::to_string(view.measureIndex));
            return false;
        }
        NativeLayoutRect pageLocal = {};
        const int page = uniqueContainingPageForRect(view.abs, &pageLocal, false);
        if (page != system->page || !isFiniteRect(pageLocal)) {
            setLayoutError(
                "BarView bbox is outside its SystemView page: measure="
                + std::to_string(view.measureIndex)
                + " bar_page=" + std::to_string(page)
                + " system_page=" + std::to_string(system->page)
                + " x=" + std::to_string(view.abs.x)
                + " y=" + std::to_string(view.abs.y)
                + " w=" + std::to_string(view.abs.w)
                + " h=" + std::to_string(view.abs.h));
            return false;
        }
        auto existing = std::find_if(
            system->measures.begin(),
            system->measures.end(),
            [&](const NativeLayoutMeasure& measure) {
                return measure.measureIndex == view.measureIndex;
            });
        if (existing == system->measures.end()) {
            system->measures.push_back({view.measureIndex, pageLocal});
        } else {
            existing->pageLocal = unionRect(existing->pageLocal, pageLocal);
        }
    }
    for (auto& system : g_layoutSystems) {
        std::sort(
            system.measures.begin(),
            system.measures.end(),
            [](const NativeLayoutMeasure& left, const NativeLayoutMeasure& right) {
                return left.measureIndex < right.measureIndex;
            });
        if (system.measures.size()
                != static_cast<size_t>(
                    system.lastMeasureIndex - system.firstMeasureIndex)) {
            std::ostringstream error;
            error << "BarView boxes do not cover SystemView range: system="
                  << system.systemIndex
                  << " range=[" << system.firstMeasureIndex << ','
                  << system.lastMeasureIndex << ") boxes="
                  << system.measures.size() << " captured=[";
            for (size_t index = 0; index < system.measures.size(); ++index) {
                if (index) error << ',';
                error << system.measures[index].measureIndex;
            }
            error << "] master_ranges=[";
            for (size_t index = 0; index < state->masterBarRanges.size(); ++index) {
                if (index) error << ',';
                error << state->masterBarRanges[index].firstBarIndex << '-'
                      << state->masterBarRanges[index].lastBarIndex
                      << (state->masterBarRanges[index].multirest ? 'm' : 'n');
            }
            error << "] multirest=" << (state->scoreViewMultiRest ? 1 : 0);
            setLayoutError(error.str());
            return false;
        }
        for (size_t measurePosition = 0;
             measurePosition < system.measures.size();
             ++measurePosition) {
            if (system.measures[measurePosition].measureIndex
                    != system.firstMeasureIndex
                        + static_cast<int>(measurePosition)) {
                setLayoutError(
                    "BarView boxes are not consecutive: system="
                    + std::to_string(system.systemIndex)
                    + " position=" + std::to_string(measurePosition)
                    + " captured="
                    + std::to_string(system.measures[measurePosition].measureIndex)
                    + " expected="
                    + std::to_string(
                        system.firstMeasureIndex
                        + static_cast<int>(measurePosition)));
                return false;
            }
        }
        std::sort(
            system.timeSignatures.begin(),
            system.timeSignatures.end(),
            [](const NativeTimeSignature& left, const NativeTimeSignature& right) {
                return std::tie(left.staffIndex, left.measureIndex)
                    < std::tie(right.staffIndex, right.measureIndex);
            });
        system.timeSignatures.erase(
            std::unique(
                system.timeSignatures.begin(),
                system.timeSignatures.end(),
                [](const NativeTimeSignature& left, const NativeTimeSignature& right) {
                    return left.staffIndex == right.staffIndex
                        && left.measureIndex == right.measureIndex
                        && left.numerator == right.numerator
                        && left.denominator == right.denominator;
                }),
            system.timeSignatures.end());
        std::sort(
            system.keySignatures.begin(),
            system.keySignatures.end(),
            [](const NativeKeySignature& left, const NativeKeySignature& right) {
                return std::tie(left.staffIndex, left.measureIndex)
                    < std::tie(right.staffIndex, right.measureIndex);
            });
        system.keySignatures.erase(
            std::unique(
                system.keySignatures.begin(),
                system.keySignatures.end(),
                [](const NativeKeySignature& left, const NativeKeySignature& right) {
                    return left.staffIndex == right.staffIndex
                        && left.measureIndex == right.measureIndex
                        && left.accidentalCount == right.accidentalCount;
                }),
            system.keySignatures.end());
    }

    if (!finalizeTempoIndications()) return false;

    const auto eventLocationKey = [](const NativeEventLocation& value) {
        return std::tie(
            value.staffIndex,
            value.measureIndex,
            value.voiceIndex,
            value.eventIndex);
    };
    if (state->restViewsResolved) {
        std::sort(
            g_restViews.begin(),
            g_restViews.end(),
            [&](const NativeRestView& left, const NativeRestView& right) {
                return eventLocationKey(left.location) < eventLocationKey(right.location);
            });
        for (size_t index = 1; index < g_restViews.size(); ++index) {
            const auto& previous = g_restViews[index - 1];
            const auto& current = g_restViews[index];
            if (eventLocationKey(previous.location) == eventLocationKey(current.location)
                    && previous.visibilityCode != current.visibilityCode) {
                state->restViewsResolved = false;
                logMsg("[LAYOUT] RestView captures conflict for one official Beat owner");
                break;
            }
        }
        if (state->restViewsResolved) {
            g_restViews.erase(
                std::unique(
                    g_restViews.begin(),
                    g_restViews.end(),
                    [&](const NativeRestView& left, const NativeRestView& right) {
                        return eventLocationKey(left.location)
                            == eventLocationKey(right.location);
                    }),
                g_restViews.end());
        } else {
            g_restViews.clear();
        }
    } else {
        g_restViews.clear();
    }
    if (state->lyricEventsResolved) {
        std::sort(
            g_lyricEvents.begin(),
            g_lyricEvents.end(),
            [&](const NativeEventLocation& left, const NativeEventLocation& right) {
                return eventLocationKey(left) < eventLocationKey(right);
            });
        g_lyricEvents.erase(
            std::unique(
                g_lyricEvents.begin(),
                g_lyricEvents.end(),
                [&](const NativeEventLocation& left, const NativeEventLocation& right) {
                    return eventLocationKey(left) == eventLocationKey(right);
                }),
            g_lyricEvents.end());
    } else {
        g_lyricEvents.clear();
    }
    return true;
}

static void noteGeometryError(const std::string& message) {
    auto* state = g_directExportState;
    if (!state) return;
    if (state->noteGeometryErrors.size() < 10
            && std::find(state->noteGeometryErrors.begin(),
                         state->noteGeometryErrors.end(), message)
                == state->noteGeometryErrors.end()) {
        state->noteGeometryErrors.push_back(message);
        logMsg("[NOTE_GEOMETRY] %s", message.c_str());
    }
}

static void finalizeNoteGeometry() {
    std::vector<NativeNoteGlyph> resolved;
    for (auto glyph : g_noteGlyphs) {
        const NativeLayoutSystem* owner = nullptr;
        for (const auto& system : g_layoutSystems) {
            if (system.firstMeasureIndex <= glyph.location.measureIndex
                    && glyph.location.measureIndex < system.lastMeasureIndex) {
                if (owner) {
                    noteGeometryError("glyph has multiple owning systems");
                    owner = nullptr;
                    break;
                }
                owner = &system;
            }
        }
        if (!owner) {
            noteGeometryError("glyph has no unique owning system");
            continue;
        }
        glyph.page = uniqueContainingPageForRect(glyph.bounds, &glyph.pageBounds, false);
        const int layoutPage = uniqueContainingPageForRect(glyph.layoutBounds, &glyph.pageLayoutBounds, false);
        if (glyph.page < 0 || glyph.page != owner->page || layoutPage != glyph.page
                || std::abs(glyph.pageBounds.w - glyph.bounds.w) > 1e-5
                || std::abs(glyph.pageBounds.h - glyph.bounds.h) > 1e-5) {
            noteGeometryError("glyph bounds lie outside their owning page");
            continue;
        }
        glyph.systemIndex = owner->systemIndex;
        const double tx = glyph.pageBounds.x - glyph.bounds.x;
        const double ty = glyph.pageBounds.y - glyph.bounds.y;
        glyph.pageOrigin = {glyph.origin.x + tx, glyph.origin.y + ty};
        resolved.push_back(std::move(glyph));
    }
    g_noteGlyphs = std::move(resolved);
}

static void writeNoteGeometry(std::ostream& out) {
    const auto& errors = g_directExportState->noteGeometryErrors;
    out << "  \"note_geometry\": {\"schema\":\"gpomr.note-glyph-geometry\","
        << "\"status\":\"" << (errors.empty() ? "complete" : "unresolved")
        << "\",\"errors\":[";
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i) out << ",";
        out << "\"" << jsonEscape(errors[i]) << "\"";
    }
    out << "],\"glyphs\":[";
    auto rect = [&](const NativeLayoutRect& r) {
        out << "[" << r.x << "," << r.y << "," << r.w << "," << r.h << "]";
    };
    for (size_t i = 0; i < g_noteGlyphs.size(); ++i) {
        const auto& glyph = g_noteGlyphs[i];
        if (i) out << ",";
        out << "{\"kind\":\"" << (glyph.noteIndex < 0 ? "rest" : "fret") << "\""
            << ",\"staff_index\":" << glyph.location.staffIndex
            << ",\"measure_index\":" << glyph.location.measureIndex
            << ",\"voice_index\":" << glyph.location.voiceIndex
            << ",\"event_index\":" << glyph.location.eventIndex
            << ",\"note_index\":";
        if (glyph.noteIndex < 0) out << "null"; else out << glyph.noteIndex;
        out << ",\"native_string_index\":";
        if (glyph.stringIndex < 0) out << "null"; else out << glyph.stringIndex;
        out << ",\"page\":" << glyph.page << ",\"system_index\":" << glyph.systemIndex
            << ",\"visibility_code\":" << glyph.visibilityCode
            << ",\"text\":";
        if (glyph.noteIndex < 0) out << "null"; else out << "\"" << jsonEscape(glyph.text) << "\"";
        out << ",\"bounds_source\":\"" << (glyph.noteIndex < 0 ? "rest_view_layout" : "native_draw_text_bounds_0") << "\""
            << ",\"bbox_mm\":";
        rect(glyph.pageBounds);
        out << ",\"layout_bbox_mm\":";
        rect(glyph.pageLayoutBounds);
        out << ",\"render_origin_mm\":[" << glyph.pageOrigin.x << "," << glyph.pageOrigin.y << "]}";
    }
    out << "]},\n";
}

static bool writeNativeLayoutDump(const std::string& path,
                                   bool pdfOk,
                                   int sourceTrackIndex) {
    if (path.empty() || !finalizeNativeLayoutSystems()) return false;
    finalizeNoteGeometry();
    QString qPath = QString::fromUtf8(path.c_str());
    QDir().mkpath(QFileInfo(qPath).absolutePath());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        logMsg("[LAYOUT] Failed to open layout output: %s", path.c_str());
        return false;
    }
    out << std::setprecision(12);

    out << "{\n";
    out << "  \"schema\": \"gpomr.render-layout\",\n";
    out << "  \"pdf_ok\": " << (pdfOk ? "true" : "false") << ",\n";
    out << "  \"source_track_index\": " << sourceTrackIndex << ",\n";
    out << "  \"tab_only\": true,\n";
    writeNoteGeometry(out);
    const auto* stylesheet = g_directExportState
        ? g_directExportState->stylesheet
        : nullptr;
    if (!stylesheet) return false;
    out << "  \"tab_style\": {"
        << "\"force_rhythmic_band\": "
        << (gp::core::style::tabBeamVisibilityValue(*stylesheet) ==
                gp::core::view::Visibility::Visible
            ? "true" : "false")
        << ", \"extend_rhythmic_in_tablature\": "
        << (static_cast<int>(
                gp::core::style::tabBeamStemAnchorValue(*stylesheet)) == 0
            ? "true" : "false")
        << ", \"hide_useless_rests\": "
        << (gp::core::style::tabStaveHideUselessRestsValue(*stylesheet)
            ? "true" : "false")
        << ", \"show_quarter_rest_as_dash\": "
        << (gp::core::style::tabStaveShowQuarterRestAsDashValue(*stylesheet)
            ? "true" : "false")
        << ", \"always_show_tie_notes\": "
        << (gp::core::style::tabStaveAlwaysShowTieNotesValue(*stylesheet)
            ? "true" : "false")
        << ", \"hide_ties_between_bar\": "
        << (gp::core::style::tabStaveHideTiesBetweenBarValue(*stylesheet)
            ? "true" : "false")
        << ", \"display_fret_relative_to_capo\": "
        << (gp::core::style::tabStaveDisplayFretRelativeToCapoValue(*stylesheet)
            ? "true" : "false")
        << ", \"fret_visibility_code\": " << static_cast<int>(
            gp::core::style::noteFretVisibilityValue(*stylesheet))
        << ", \"trill_fret_visibility_code\": " << static_cast<int>(
            gp::core::style::noteFretTrillVisibilityValue(*stylesheet))
        << ", \"artificial_harmonic_fret_visibility_code\": " << static_cast<int>(
            gp::core::style::noteFretArtificialHarmonicVisibilityValue(*stylesheet))
        << "},\n";
    out << "  \"tuning_style\": {"
        << "\"position_code\": " << static_cast<int>(
            gp::core::style::scoreTuningPositionValue(*stylesheet))
        << ", \"mode_code\": " << static_cast<int>(
            gp::core::style::scoreTuningModeValue(*stylesheet))
        << ", \"column_count\": "
        << gp::core::style::scoreTuningColumnCountValue(*stylesheet)
        << ", \"boxed\": "
        << (gp::core::style::scoreTuningBoxedValue(*stylesheet)
            ? "true" : "false")
        << "},\n";
    out << "  \"pages\": [\n";
    for (size_t i = 0; i < g_layoutPages.size(); ++i) {
        const auto& p = g_layoutPages[i];
        out << "    {\"index\": " << p.index
            << ", \"bbox_mm\": [0, 0, "
            << p.rect.w << ", " << p.rect.h << "]}";
        out << (i + 1 == g_layoutPages.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"systems\": [\n";
    for (size_t i = 0; i < g_layoutSystems.size(); ++i) {
        const auto& system = g_layoutSystems[i];
        out << "    {\"system_index\": " << system.systemIndex
            << ", \"page\": " << system.page
            << ", \"first_measure_index\": " << system.firstMeasureIndex
            << ", \"last_measure_index\": " << system.lastMeasureIndex
            << ", \"bbox_mm\": [" << system.pageLocal.x << ", "
            << system.pageLocal.y << ", " << system.pageLocal.w << ", "
            << system.pageLocal.h << "]"
            << ", \"track_label\": ";
        if (system.trackLabel.empty()) {
            out << "null";
        } else {
            out << "\"" << jsonEscape(system.trackLabel) << "\"";
        }
        out << ", \"time_signatures\": [";
        for (size_t signatureIndex = 0;
             signatureIndex < system.timeSignatures.size();
             ++signatureIndex) {
            const auto& signature = system.timeSignatures[signatureIndex];
            if (signatureIndex) out << ", ";
            out << "{\"staff_index\": " << signature.staffIndex
                << ", \"measure_index\": " << signature.measureIndex
                << ", \"numerator\": " << signature.numerator
                << ", \"denominator\": " << signature.denominator << "}";
        }
        out << "], \"measure_boxes\": [";
        for (size_t measureIndex = 0;
             measureIndex < system.measures.size();
             ++measureIndex) {
            const auto& measure = system.measures[measureIndex];
            if (measureIndex) out << ", ";
            out << "{\"measure_index\": " << measure.measureIndex
                << ", \"bbox_mm\": [" << measure.pageLocal.x << ", "
                << measure.pageLocal.y << ", " << measure.pageLocal.w << ", "
                << measure.pageLocal.h << "]}";
        }
        out << "], \"key_signatures\": [";
        for (size_t signatureIndex = 0;
             signatureIndex < system.keySignatures.size();
             ++signatureIndex) {
            const auto& signature = system.keySignatures[signatureIndex];
            if (signatureIndex) out << ", ";
            out << "{\"staff_index\": " << signature.staffIndex
                << ", \"measure_index\": " << signature.measureIndex
                << ", \"accidental_count\": "
                << signature.accidentalCount << "}";
        }
        out << "]}";
        out << (i + 1 == g_layoutSystems.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"tempo_indications\": [\n";
    for (size_t i = 0; i < g_tempoIndications.size(); ++i) {
        const auto& indication = g_tempoIndications[i];
        out << "    {\"page\": " << indication.page
            << ", \"system_index\": ";
        if (indication.initial) {
            out << "null";
        } else {
            out << indication.systemIndex;
        }
        out << ", \"master_measure_index\": "
            << indication.masterMeasureIndex
            << ", \"position\": " << indication.position
            << ", \"bbox_mm\": [" << indication.pageLocal.x << ", "
            << indication.pageLocal.y << ", " << indication.pageLocal.w << ", "
            << indication.pageLocal.h << "]"
            << ", \"source\": \""
            << (indication.initial ? "initial" : "automation") << "\"}";
        out << (i + 1 == g_tempoIndications.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"rest_views\": ";
    if (!g_directExportState->restViewsResolved) {
        out << "null";
    } else {
        out << "[";
        for (size_t viewIndex = 0; viewIndex < g_restViews.size(); ++viewIndex) {
            const auto& view = g_restViews[viewIndex];
            if (viewIndex) out << ", ";
            out << "{\"staff_index\": " << view.location.staffIndex
                << ", \"measure_index\": " << view.location.measureIndex
                << ", \"voice_index\": " << view.location.voiceIndex
                << ", \"event_index\": " << view.location.eventIndex
                << ", \"visibility_code\": " << view.visibilityCode << "}";
        }
        out << "]";
    }
    out << ",\n";
    out << "  \"lyric_events\": ";
    if (!g_directExportState->lyricEventsResolved) {
        out << "null";
    } else {
        out << "[";
        for (size_t eventIndex = 0; eventIndex < g_lyricEvents.size(); ++eventIndex) {
            const auto& event = g_lyricEvents[eventIndex];
            if (eventIndex) out << ", ";
            out << "{\"staff_index\": " << event.staffIndex
                << ", \"measure_index\": " << event.measureIndex
                << ", \"voice_index\": " << event.voiceIndex
                << ", \"event_index\": " << event.eventIndex << "}";
        }
        out << "]";
    }
    out << "\n";
    out << "}\n";
    out.flush();
    logMsg(
        "[LAYOUT] Wrote %zu systems to %s",
        g_layoutSystems.size(),
        path.c_str());
    return out.good();
}

static bool callViewBoundingRect(void* view, NativeLayoutRect* out) {
    if (!out || !view || !g_viewBoundingRect) return false;
    __try {
        const auto& rect = g_viewBoundingRect(view);
        NativeLayoutRect result = {rect.x, rect.y, rect.w, rect.h};
        if (!isFiniteRect(result)) return false;
        *out = result;
        return true;
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

static bool hasBarViewElement(fn_BarViewElement getter, const void* view) {
    if (!getter || !view) return false;
    __try {
        return static_cast<bool>(getter(view));
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

using fn_OnRender = void (__fastcall *)(
    void*,
    void*,
    const am::painting::Point*,
    const am::painting::Rect*);

static void __fastcall hooked_BarViewOnRender(
        void* self,
        void* painter,
        const am::painting::Point* offset,
        const am::painting::Rect* clip) {
    auto original = reinterpret_cast<fn_OnRender>(g_layoutBarViewHook.trampoline);
    if (!g_layoutDumpActive || !self) {
        original(self, painter, offset, clip);
        return;
    }

    const auto* barView = reinterpret_cast<const gp::core::view::BarView*>(self);
    const auto bar = barView->model();
    DirectExportState* state = g_directExportState;
    if (!bar || !state || state->sourceTrackIndex < 0) {
        original(self, painter, offset, clip);
        return;
    }

    const int measureIndex = static_cast<int>(bar->index());
    const int staffIndex = static_cast<int>(bar->staffIndex());
    const auto masterBar = bar->masterBar();
    if (!masterBar) {
        original(self, painter, offset, clip);
        return;
    }
    if (offset) {
        const NativeLayoutRect abs = translateRect(
            rectFromView(self), {offset->x, offset->y});
        if (isFiniteRect(abs)) {
            g_layoutMeasureViews.push_back({staffIndex, measureIndex, abs});
        }
    }
    if (hasBarViewElement(g_barViewTimeSignatureElement, self)) {
        const auto& signature = masterBar->timeSignature();
        g_timeSignatures.push_back({
            staffIndex,
            measureIndex,
            signature.getNumerator(),
            signature.getDenominator(),
        });
    }
    if (hasBarViewElement(g_barViewKeySignatureElement, self)) {
        const auto signature = masterBar->drawingKeySignatureOfTrackAndStaff(
            static_cast<unsigned int>(state->sourceTrackIndex),
            static_cast<unsigned int>(staffIndex));
        g_keySignatures.push_back({
            staffIndex,
            measureIndex,
            signature.accidentalCount(),
        });
    }
    original(self, painter, offset, clip);
}

using fn_SystemViewOnRender = fn_OnRender;

static bool originalBarRangeForSystem(
        const gp::core::view::generated::SystemViewBase& system,
        int* firstBarIndex,
        int* lastBarIndex) {
    DirectExportState* state = g_directExportState;
    if (!state || !firstBarIndex || !lastBarIndex) return false;
    const int firstMasterBarIndex = system.firstMasterBarIndex();
    const int masterBarCount = system.masterBarCount();
    if (firstMasterBarIndex < 0 || masterBarCount <= 0) return false;
    const size_t first = static_cast<size_t>(firstMasterBarIndex);
    const size_t end = first + static_cast<size_t>(masterBarCount);
    if (first >= state->masterBarRanges.size()
            || end > state->masterBarRanges.size()) {
        return false;
    }
    *firstBarIndex = state->masterBarRanges[first].firstBarIndex;
    *lastBarIndex = state->masterBarRanges[end - 1].lastBarIndex;
    return *firstBarIndex >= 0 && *lastBarIndex > *firstBarIndex;
}

static bool isOwnedSystemAttachment(
        const std::shared_ptr<gp::core::view::SystemAttachedView>& attached,
        const gp::core::view::SystemView* expectedSystem) {
    if (!attached || !expectedSystem) return false;
    const auto owner = attached->systemView();
    if (!owner || owner.get() != expectedSystem) return false;
    const auto& element = attached->element();
    return static_cast<bool>(element);
}

static void captureSystemAttachments(
        const gp::core::view::generated::SystemViewBase& system,
        const gp::core::view::SystemView* systemView,
        NativeLayoutSystem* layoutSystem) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->stylesheet || !layoutSystem) return;

    const auto labelMode = system.firstMasterBarIndex() == 0
        ? gp::core::style::scoreSystemsFirstSystemLabelModeValue(
            *state->stylesheet)
        : gp::core::style::scoreSystemsNextSystemsLabelModeValue(
            *state->stylesheet);
    const auto& labels = system.systemLabels();
    const char* trackLabel = nullptr;
    switch (labelMode) {
        case gp::core::style::generated::SystemLabelStyle::Mode::Full:
            trackLabel = "name";
            break;
        case gp::core::style::generated::SystemLabelStyle::Mode::Abbreviated:
            trackLabel = "short_name";
            break;
        case gp::core::style::generated::SystemLabelStyle::Mode::None:
            break;
        default:
            break;
    }

    if (trackLabel && std::any_of(
            labels.begin(),
            labels.end(),
            [&](const auto& label) {
                return isOwnedSystemAttachment(label, systemView);
            })) {
        layoutSystem->trackLabel = trackLabel;
    }
}

static void __fastcall hooked_SystemViewOnRender(void* self,
                                                  void* painter,
                                                  const am::painting::Point* offset,
                                                  const am::painting::Rect* clip) {
    auto original = reinterpret_cast<fn_SystemViewOnRender>(
        g_layoutSystemViewHook.trampoline);
    if (!g_layoutDumpActive) {
        original(self, painter, offset, clip);
        return;
    }
    if (self && offset) {
        const auto* system = reinterpret_cast<
            const gp::core::view::generated::SystemViewBase*>(self);
        NativeLayoutSystem layoutSystem;
        const bool hasBarRange = originalBarRangeForSystem(
            *system,
            &layoutSystem.firstMeasureIndex,
            &layoutSystem.lastMeasureIndex);
        const int previousSystemFirstMeasureIndex =
            g_renderedSystemFirstMeasureIndex;
        g_renderedSystemFirstMeasureIndex = hasBarRange
            ? layoutSystem.firstMeasureIndex
            : -1;
        original(self, painter, offset, clip);
        g_renderedSystemFirstMeasureIndex = previousSystemFirstMeasureIndex;
        if (g_directExportState
                && !g_directExportState->layoutError.empty()) {
            return;
        }
        if (hasBarRange) {
            layoutSystem.abs = translateRect(
                rectFromView(self), {offset->x, offset->y});
            captureSystemAttachments(
                *system,
                reinterpret_cast<const gp::core::view::SystemView*>(self),
                &layoutSystem);
            addLayoutSystem(std::move(layoutSystem));
        } else {
            setLayoutError("SystemView has an invalid MasterBarView slice");
        }
        return;
    }

    original(self, painter, offset, clip);
}









static bool sameEventLocation(
        const NativeEventLocation& left,
        const NativeEventLocation& right) {
    return left.staffIndex == right.staffIndex
        && left.measureIndex == right.measureIndex
        && left.voiceIndex == right.voiceIndex
        && left.eventIndex == right.eventIndex;
}

static void markRestViewsUnresolved(const char* reason) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->restViewsResolved) return;
    state->restViewsResolved = false;
    g_restViews.clear();
    logMsg("[LAYOUT] RestView gate unresolved: %s", reason);
}

static void markLyricEventsUnresolved(const char* reason) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->lyricEventsResolved) return;
    state->lyricEventsResolved = false;
    g_lyricEvents.clear();
    logMsg("[LAYOUT] Lyric view gate unresolved: %s", reason);
}

static void collectSelectedTrackLocations(const gp::core::Track& track) {
    DirectExportState* state = g_directExportState;
    if (!state) return;
    state->beatLocations.clear();
    state->restViewsResolved = true;
    state->lyricEventsResolved = true;

    for (unsigned int staffPosition = 0;
         staffPosition < track.staffCount();
         ++staffPosition) {
        const auto staff = track.staff(staffPosition);
        if (!staff) {
            markRestViewsUnresolved("selected Track contains an empty Staff");
            markLyricEventsUnresolved("selected Track contains an empty Staff");
            return;
        }
        for (unsigned int barPosition = 0;
             barPosition < staff->barCount();
             ++barPosition) {
            const auto bar = staff->bar(barPosition);
            if (!bar) {
                markRestViewsUnresolved("selected Staff contains an empty Bar");
                markLyricEventsUnresolved("selected Staff contains an empty Bar");
                return;
            }
            for (unsigned int voicePosition = 0;
                 voicePosition < bar->voiceCount();
                 ++voicePosition) {
                const auto voice = bar->voice(voicePosition);
                if (!voice) {
                    markRestViewsUnresolved("selected Bar contains an empty Voice");
                    markLyricEventsUnresolved("selected Bar contains an empty Voice");
                    return;
                }
                for (unsigned int beatPosition = 0;
                     beatPosition < voice->beatCount();
                     ++beatPosition) {
                    const auto beat = voice->beat(beatPosition);
                    if (!beat) {
                        markRestViewsUnresolved("selected Voice contains an empty Beat");
                        markLyricEventsUnresolved("selected Voice contains an empty Beat");
                        return;
                    }
                    const NativeEventLocation eventLocation = {
                        static_cast<int>(staff->index()),
                        static_cast<int>(bar->index()),
                        static_cast<int>(voice->index()),
                        static_cast<int>(beat->index()),
                    };
                    const auto beatInserted = state->beatLocations.emplace(
                        beat.get(), eventLocation);
                    if (!beatInserted.second
                            && !sameEventLocation(
                                beatInserted.first->second, eventLocation)) {
                        markRestViewsUnresolved(
                            "one Beat has multiple official locations");
                        markLyricEventsUnresolved(
                            "one Beat has multiple official locations");
                        return;
                    }
                }
            }
        }
    }
}

static void captureLyricBandElement(const void* self) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->lyricEventsResolved || !self) return;
    if (!g_bandElementViewType || !g_lyricBandElementAttachedBeat
            || !g_viewVisibility) {
        markLyricEventsUnresolved("official lyric View getters are unavailable");
        return;
    }
    try {
        if (g_bandElementViewType(self)
                != gp::core::view::BandElementType::Lyrics) return;
        const int visibilityCode = static_cast<int>(g_viewVisibility(self));
        if (visibilityCode < 0 || visibilityCode > 2) {
            markLyricEventsUnresolved("LyricBandElement has an invalid visibility");
            return;
        }
        if (visibilityCode != static_cast<int>(
                gp::core::view::Visibility::Visible)) {
            return;
        }
        const auto& beat = g_lyricBandElementAttachedBeat(self);
        if (!beat) {
            markLyricEventsUnresolved("LyricBandElement has no Beat owner");
            return;
        }
        const auto beatLocation = state->beatLocations.find(beat.get());
        if (beatLocation == state->beatLocations.end()) {
            markLyricEventsUnresolved(
                "LyricBandElement has no exact selected-track Beat owner");
            return;
        }
        g_lyricEvents.push_back(beatLocation->second);
    } catch (...) {
        markLyricEventsUnresolved("official lyric owner getter failed");
    }
}

static void captureRestView(const void* self) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->restViewsResolved || !self) return;
    if (!g_noteViewBeatModel || !g_noteViewVoiceIndex || !g_viewVisibility) {
        markRestViewsUnresolved("official View getters are unavailable");
        return;
    }
    try {
        std::shared_ptr<gp::core::Beat> beat;
        g_noteViewBeatModel(self, &beat);
        if (!beat) return;
        const auto beatLocation = state->beatLocations.find(beat.get());
        if (beatLocation == state->beatLocations.end()
                || g_noteViewVoiceIndex(self) != beatLocation->second.voiceIndex) {
            markRestViewsUnresolved(
                "RestView has no exact selected-track Beat owner");
            return;
        }
        const int visibilityCode = static_cast<int>(g_viewVisibility(self));
        if (visibilityCode < 0 || visibilityCode > 2) {
            markRestViewsUnresolved("RestView has an invalid visibility");
            return;
        }
        g_restViews.push_back({
            beatLocation->second,
            visibilityCode,
        });
    } catch (...) {
        markRestViewsUnresolved("official owner getter failed");
    }
}

static bool captureTempoBandElement(
        const void* self,
        const am::painting::Point* offset) {
    DirectExportState* state = g_directExportState;
    if (!state || !state->tempoIndicationsResolved || !self
            || !g_bandElementViewType) return false;
    try {
        if (g_bandElementViewType(self)
                != gp::core::view::BandElementType::Tempo) return false;
        if (!offset || g_renderedSystemFirstMeasureIndex < 0) {
            markTempoIndicationsUnresolved(
                "tempo BandElementView has no owning SystemView render");
            return true;
        }
        if (std::any_of(
                g_tempoViewCaptures.begin(),
                g_tempoViewCaptures.end(),
                [&](const NativeTempoViewCapture& capture) {
                    return capture.identity == self;
                })) return true;
        const NativeLayoutRect abs = translateRect(
            rectFromView(const_cast<void*>(self)), {offset->x, offset->y});
        if (!isFiniteRect(abs)) {
            markTempoIndicationsUnresolved(
                "tempo BandElementView has an invalid official bounding box");
            return true;
        }
        g_tempoViewCaptures.push_back({
            self,
            g_renderedSystemFirstMeasureIndex,
            abs,
        });
        return true;
    } catch (...) {
        markTempoIndicationsUnresolved(
            "official tempo BandElementView getter failed");
        return true;
    }
}

static void captureNoteGlyph(const void* self, void* painter, const am::painting::Point* offset, bool isRest) {
    auto* state = g_directExportState;
    if (!state || !state->captureNoteGeometry || !g_noteGeometryHookReady.load() || !self || !offset) return;
    try {
        if (!painter) { noteGeometryError("glyph has no renderer painter"); return; }
        auto** vtable = *reinterpret_cast<void***>(painter);
        auto nativePainter = reinterpret_cast<void* (__fastcall *)(void*)>(vtable[15])(painter);
        if (!nativePainter || !currentPainterTranslation(nativePainter).translationOnly) {
            noteGeometryError("glyph uses an unsupported non-translation painter transform");
            return;
        }
        const int visibility = static_cast<int>(g_viewVisibility(self));
        if (visibility != static_cast<int>(gp::core::view::Visibility::Visible)) return;
        std::shared_ptr<gp::core::Beat> beat;
        std::shared_ptr<gp::core::Note> note;
        gp::core::Beat* ownerBeat = nullptr;
        if (isRest) {
            g_noteViewBeatModel(self, &beat);
            ownerBeat = beat.get();
        } else {
            g_noteViewNoteModel(self, &note);
            if (note) ownerBeat = g_noteParentBeat(note.get());
        }
        if (!ownerBeat) { noteGeometryError("drawn glyph has no Beat owner"); return; }
        const auto location = state->beatLocations.find(ownerBeat);
        if (location == state->beatLocations.end()
                || location->second.voiceIndex != g_noteViewVoiceIndex(self)) {
            noteGeometryError("drawn glyph has no exact selected-track event owner");
            return;
        }
        NativeNoteGlyph glyph;
        glyph.location = location->second;
        glyph.origin = {offset->x, offset->y};
        glyph.visibilityCode = visibility;
        glyph.layoutBounds = translateRect(rectFromView(const_cast<void*>(self)), glyph.origin);
        if (isRest) {
            glyph.bounds = glyph.layoutBounds;
        } else {
            bool owned = false;
            for (unsigned int i = 0; note && i < ownerBeat->noteCount(); ++i) {
                if (ownerBeat->note(i).get() == note.get()) owned = true;
            }
            if (!owned) { noteGeometryError("fret glyph Note is not owned by its Beat"); return; }
            glyph.noteIndex = static_cast<int>(note->index());
            glyph.stringIndex = static_cast<int>(note->string());
            glyph.text = g_fretViewText(self);
            if (glyph.text.empty()) return;
            am::painting::Rect bounds;
            g_styleTextBounds(state->stylesheet, &bounds, &glyph.text, g_fretViewFont(self), 2, 2, 0);
            glyph.bounds = translateRect({bounds.x, bounds.y, bounds.w, bounds.h}, glyph.origin);
        }
        if (!isFiniteRect(glyph.bounds) || !isFiniteRect(glyph.layoutBounds)) {
            noteGeometryError("glyph has invalid draw or layout bounds");
            return;
        }
        g_noteGlyphs.push_back(std::move(glyph));
    } catch (...) { noteGeometryError("native glyph owner or bounds getter failed"); }
}

using fn_FretOnRender = void (__fastcall *)(void*, void*, const am::painting::Point*, const am::painting::Rect*, const void*);
static void __fastcall hooked_FretViewOnRender(void* self, void* painter,
        const am::painting::Point* offset, const am::painting::Rect* clip, const void* color) {
    if (g_layoutDumpActive) captureNoteGlyph(self, painter, offset, false);
    reinterpret_cast<fn_FretOnRender>(g_layoutFretViewHook.trampoline)(self, painter, offset, clip, color);
}

static void __fastcall hooked_RestViewOnRender(
        void* self,
        void* painter,
        const am::painting::Point* offset,
    const am::painting::Rect* clip) {
    auto original = reinterpret_cast<fn_OnRender>(g_layoutRestViewHook.trampoline);
    if (g_layoutDumpActive) {
        captureRestView(self);
        captureNoteGlyph(self, painter, offset, true);
    }
    original(self, painter, offset, clip);
}

static void __fastcall hooked_BandElementViewOnRender(
        void* self,
        void* painter,
        const am::painting::Point* offset,
        const am::painting::Rect* clip) {
    auto original = reinterpret_cast<fn_OnRender>(
        g_layoutBandElementViewHook.trampoline);
    bool tempo = false;
    if (g_layoutDumpActive) {
        captureLyricBandElement(self);
        tempo = captureTempoBandElement(self, offset);
    }
    if (tempo) ++g_tempoBandElementRenderDepth;
    original(self, painter, offset, clip);
    if (tempo) --g_tempoBandElementRenderDepth;
}

static void removeNativeLayoutHooks();

static bool installNativeLayoutHooks() {
    std::call_once(g_layoutHooksOnce, []() {
        HMODULE gpcore = GetModuleHandleA("GPCore.dll");
        HMODULE ampainting = GetModuleHandleA("AMPainting.dll");
        if (!gpcore || !ampainting) return;
        auto paintingRva = [&](uintptr_t rva) -> void* {
            return reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(ampainting) + rva);
        };

        g_layoutSystemViewHook.target = exportedFunction(
            gpcore,
            "?onRender@SystemView@view@core@gp@@UEBAXAEAVIPainter@renderer@34@"
            "AEBVPoint@painting@am@@AEBVRect@89@@Z");
        g_layoutSystemViewHook.detour = reinterpret_cast<void*>(&hooked_SystemViewOnRender);
        g_layoutSystemViewHook.patchSize = 21;
        g_layoutSystemViewHook.createTrampoline = true;

        g_layoutBarViewHook.target = exportedFunction(
            gpcore,
            "?onRender@BarView@view@core@gp@@UEBAXAEAVIPainter@renderer@34@"
            "AEBVPoint@painting@am@@AEBVRect@89@@Z");
        g_layoutBarViewHook.detour = reinterpret_cast<void*>(
            &hooked_BarViewOnRender);
        g_layoutBarViewHook.patchSize = 15;
        g_layoutBarViewHook.createTrampoline = true;

        g_layoutRestViewHook.target = exportedFunction(
            gpcore,
            "?onRender@RestView@view@core@gp@@UEBAXAEAVIPainter@renderer@34@"
            "AEBVPoint@painting@am@@AEBVRect@89@@Z");
        g_layoutRestViewHook.detour = reinterpret_cast<void*>(&hooked_RestViewOnRender);
        g_layoutRestViewHook.patchSize = 15;
        g_layoutRestViewHook.createTrampoline = true;

        g_layoutFretViewHook.target = exportedFunction(gpcore,
            "?onRender@FretView@view@core@gp@@UEBAXAEAVIPainter@renderer@34@AEBVPoint@painting@am@@AEBVRect@89@AEBVColor@89@@Z");
        g_layoutFretViewHook.detour = reinterpret_cast<void*>(&hooked_FretViewOnRender);
        g_layoutFretViewHook.patchSize = 15;
        g_layoutFretViewHook.createTrampoline = true;
        g_noteViewNoteModel = reinterpret_cast<fn_NoteViewNoteModel>(exportedFunction(gpcore,
            "?noteModel@NoteView@view@core@gp@@QEBA?AV?$shared_ptr@VNote@core@gp@@@std@@XZ"));
        g_noteParentBeat = reinterpret_cast<fn_NoteParentBeat>(exportedFunction(gpcore,
            "?parentBeat@Note@core@gp@@QEBAPEAVBeat@23@XZ"));
        g_fretViewText = reinterpret_cast<fn_FretViewText>(exportedFunction(gpcore,
            "?text@FretViewBase@generated@view@core@gp@@QEBAAEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@XZ"));
        g_fretViewFont = reinterpret_cast<fn_FretViewFont>(exportedFunction(gpcore,
            "?font@FretViewBase@generated@view@core@gp@@QEBAAEBVFont@painting@am@@XZ"));
        g_styleTextBounds = reinterpret_cast<fn_StyleTextBounds>(exportedFunction(gpcore,
            "?textBoundingRect@Stylesheet@style@core@gp@@QEBA?AVRect@painting@am@@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBVFont@67@W4TextHAlign@67@W4TextVAlign@67@W4TextBoundsType@67@@Z"));

        g_layoutBandElementViewHook.target = exportedFunction(
            gpcore,
            "?onRender@BandElementView@view@core@gp@@UEBAXAEAVIPainter@renderer@34@"
            "AEBVPoint@painting@am@@AEBVRect@89@@Z");
        g_layoutBandElementViewHook.detour = reinterpret_cast<void*>(
            &hooked_BandElementViewOnRender);
        g_layoutBandElementViewHook.patchSize = 15;
        g_layoutBandElementViewHook.createTrampoline = true;

        g_layoutTempoIndicationRenderHook.target = paintingRva(0x733C0);
        g_layoutTempoIndicationRenderHook.detour = reinterpret_cast<void*>(
            &hookedTempoIndicationRender);
        g_layoutTempoIndicationRenderHook.patchSize = 14;
        g_layoutTempoIndicationRenderHook.createTrampoline = false;

        g_layoutPaintDevicePushMatrixHook.target = paintingRva(0x77830);
        g_layoutPaintDevicePushMatrixHook.detour = reinterpret_cast<void*>(
            &hookedPaintDevicePushMatrix);
        g_layoutPaintDevicePushMatrixHook.patchSize = 15;
        g_layoutPaintDevicePushMatrixHook.createTrampoline = true;

        g_layoutPaintDevicePopMatrixHook.target = paintingRva(0x77510);
        g_layoutPaintDevicePopMatrixHook.detour = reinterpret_cast<void*>(
            &hookedPaintDevicePopMatrix);
        g_layoutPaintDevicePopMatrixHook.patchSize = 14;
        g_layoutPaintDevicePopMatrixHook.createTrampoline = false;

        g_viewBoundingRect = reinterpret_cast<fn_ViewBoundingRect>(
            exportedFunction(
                gpcore,
                "?boundingRect@ViewBase@generated@view@core@gp@@QEBAAEBVRect@"
                "painting@am@@XZ"));
        g_noteViewBeatModel = reinterpret_cast<fn_NoteViewBeatModel>(
            exportedFunction(
                gpcore,
                "?beatModel@NoteView@view@core@gp@@QEBA?AV?$shared_ptr@"
                "VBeat@core@gp@@@std@@XZ"));
        g_noteViewVoiceIndex = reinterpret_cast<fn_NoteViewVoiceIndex>(
            exportedFunction(
                gpcore,
                "?voiceIndex@NoteView@view@core@gp@@QEBAHXZ"));
        g_viewVisibility = reinterpret_cast<fn_ViewVisibility>(
            exportedFunction(
                gpcore,
                "?visibility@ViewBase@generated@view@core@gp@@"
                "QEBA?AW4Visibility@345@XZ"));
        g_bandElementViewType = reinterpret_cast<fn_BandElementViewType>(
            exportedFunction(
                gpcore,
                "?type@BandElementViewBase@generated@view@core@gp@@"
                "QEBA?AW4BandElementType@345@XZ"));
        g_lyricBandElementAttachedBeat =
            reinterpret_cast<fn_LyricBandElementAttachedBeat>(
                exportedFunction(
                    gpcore,
                    "?attachedBeat@LyricBandElement@view@core@gp@@"
                    "QEBAAEBV?$shared_ptr@$$CBVBeat@core@gp@@@std@@XZ"));
        g_offscreenDrawingElementRender =
            reinterpret_cast<fn_TempoIndicationRender>(
                exportedFunction(
                    ampainting,
                    "?render@OffscreenDrawingElement@painting@am@@"
                    "UEBAXAEAVPainter@23@AEBV?$optional@VPen@painting@am@@@std@@"
                    "AEBV?$optional@VBrush@painting@am@@@6@@Z"));
        g_matrixTx = reinterpret_cast<fn_MatrixValue>(
            exportedFunction(
                ampainting,
                "?tx@Matrix@painting@am@@QEBANXZ"));
        g_matrixTy = reinterpret_cast<fn_MatrixValue>(
            exportedFunction(
                ampainting,
                "?ty@Matrix@painting@am@@QEBANXZ"));
        g_matrixIsTranslation = reinterpret_cast<fn_MatrixIsTranslation>(
            exportedFunction(
                ampainting,
                "?isTranslation@Matrix@painting@am@@QEBA_NXZ"));
        g_barViewTimeSignatureElement = reinterpret_cast<fn_BarViewElement>(
            exportedFunction(
                gpcore,
                "?timeSignatureElement@BarViewBase@generated@view@core@gp@@"
                "QEBAAEBV?$shared_ptr@VIElement@painting@am@@@std@@XZ"));
        g_barViewKeySignatureElement = reinterpret_cast<fn_BarViewElement>(
            exportedFunction(
                gpcore,
                "?keySignatureElement@BarViewBase@generated@view@core@gp@@"
                "QEBAAEBV?$shared_ptr@VIElement@painting@am@@@std@@XZ"));

        const bool requiredCapabilitiesAvailable = g_layoutSystemViewHook.target
            && g_layoutBarViewHook.target
            && g_layoutBandElementViewHook.target
            && g_layoutTempoIndicationRenderHook.target
            && g_layoutPaintDevicePushMatrixHook.target
            && g_layoutPaintDevicePopMatrixHook.target
            && g_viewBoundingRect
            && g_bandElementViewType
            && g_offscreenDrawingElementRender
            && g_matrixTx
            && g_matrixTy
            && g_matrixIsTranslation
            && g_barViewTimeSignatureElement
            && g_barViewKeySignatureElement;
        const bool ok = requiredCapabilitiesAvailable
            && installInlineHook(&g_layoutSystemViewHook)
            && installInlineHook(&g_layoutBarViewHook)
            && installInlineHook(&g_layoutBandElementViewHook)
            && installInlineHook(&g_layoutTempoIndicationRenderHook)
            && installInlineHook(&g_layoutPaintDevicePushMatrixHook)
            && installInlineHook(&g_layoutPaintDevicePopMatrixHook);
        if (!ok) {
            logMsg(
                "[LAYOUT] Required native page/system/bar/tempo views are unavailable");
            removeNativeLayoutHooks();
        }
        const bool restOk = ok
            && g_layoutRestViewHook.target
            && g_noteViewBeatModel
            && g_noteViewVoiceIndex
            && g_viewVisibility
            && installInlineHook(&g_layoutRestViewHook);
        const bool lyricOk = ok
            && g_lyricBandElementAttachedBeat
            && g_viewVisibility;
        if (ok && !restOk) {
            logMsg("[LAYOUT] Optional RestView gate is unavailable");
        }
        if (ok && !lyricOk) {
            logMsg("[LAYOUT] Optional lyric View gate is unavailable");
        }
        g_restViewHookReady.store(restOk);
        // Three complete MOV instructions; reject an unknown binary prologue.
        const BYTE fretPrologue[] = {0x48,0x89,0x5c,0x24,0x08,0x48,0x89,0x6c,0x24,0x10,0x48,0x89,0x74,0x24,0x18};
        const bool geometryOk = restOk && g_noteViewNoteModel && g_noteParentBeat && g_fretViewText && g_fretViewFont && g_styleTextBounds
            && g_layoutFretViewHook.target
            && memcmp(g_layoutFretViewHook.target, fretPrologue, sizeof(fretPrologue)) == 0
            && installInlineHook(&g_layoutFretViewHook);
        g_noteGeometryHookReady.store(geometryOk);
        logMsg("[NOTE_GEOMETRY] native FretView capture available=%d", geometryOk ? 1 : 0);
        g_lyricBandElementHookReady.store(lyricOk);
        g_tempoIndicationHooksReady.store(ok);
        g_layoutHooksReady.store(ok);
        logMsg(
            "[LAYOUT] Native hooks installed required=%d rest=%d lyric=%d tempo=%d",
            ok ? 1 : 0,
            restOk ? 1 : 0,
            lyricOk ? 1 : 0,
            ok ? 1 : 0);
    });
    return g_layoutHooksReady.load();
}

static void removeNativeLayoutHooks() {
    removeInlineHook(&g_layoutFretViewHook);
    removeInlineHook(&g_layoutPaintDevicePopMatrixHook);
    removeInlineHook(&g_layoutPaintDevicePushMatrixHook);
    removeInlineHook(&g_layoutTempoIndicationRenderHook);
    removeInlineHook(&g_layoutRestViewHook);
    removeInlineHook(&g_layoutBandElementViewHook);
    removeInlineHook(&g_layoutBarViewHook);
    removeInlineHook(&g_layoutSystemViewHook);
}

static bool collectOriginalMasterBarRanges(
        const gp::core::ScoreView& view,
        std::vector<NativeMasterBarRange>* ranges,
        std::string& errOut) {
    if (!ranges) return false;
    ranges->clear();
    int nextBarIndex = 0;
    for (const auto& masterBar : view.masterBars()) {
        if (!masterBar) {
            errOut = "Guitar Pro ScoreView contains an empty MasterBarView";
            return false;
        }
        NativeMasterBarRange range = {
            masterBar->firstBarIndex(),
            masterBar->lastBarIndex(),
            masterBar->isMultirest(),
        };
        if (range.firstBarIndex != nextBarIndex
                || range.lastBarIndex <= range.firstBarIndex) {
            errOut = "Guitar Pro ScoreView has invalid MasterBarView ranges";
            return false;
        }
        ranges->push_back(range);
        nextBarIndex = range.lastBarIndex;
    }
    if (ranges->empty()) {
        errOut = "Guitar Pro ScoreView contains no MasterBarViews";
        return false;
    }
    return true;
}

static bool collectTempoOwners(
        const gp::core::MasterTrack& masterTrack,
        std::vector<NativeTempoOwner>* owners,
        std::string& errOut) {
    if (!owners) return false;
    owners->clear();
    try {
        const unsigned int measureCount = masterTrack.masterBarCount();
        if (masterTrack.tempoVisible()) {
            if (measureCount == 0) {
                errOut = "visible initial tempo has no master measure owner";
                return false;
            }
            owners->push_back({0, 0.0, true});
        }

        std::vector<std::shared_ptr<gp::core::Automation>> automations;
        masterTrack.gp::core::AutomationContainerProxy::getAutomations(
            automations);
        const bool initialTempoVisible = masterTrack.tempoVisible();
        std::set<std::pair<unsigned int, float>> automationLocations;
        for (const auto& automation : automations) {
            if (!automation
                    || gp::core::Automation::typeToString(automation->type())
                        != "Tempo"
                    || !automation->isVisible()) {
                continue;
            }
            const unsigned int measureIndex =
                automation->gp::core::Automation::barIndex();
            const float position = automation->position();
            if (measureIndex >= measureCount
                    || !std::isfinite(position)
                    || position < 0.0f
                    || position > 1.0f) {
                errOut = "visible tempo automation has an invalid official owner";
                return false;
            }
            if (initialTempoVisible && measureIndex == 0 && position == 0.0f) {
                continue;
            }
            if (!automationLocations.emplace(measureIndex, position).second) {
                errOut = "visible tempo automation owners are not unique";
                return false;
            }
            owners->push_back({
                static_cast<int>(measureIndex),
                static_cast<double>(position),
                false,
            });
        }
        std::sort(
            owners->begin(),
            owners->end(),
            [](const NativeTempoOwner& left, const NativeTempoOwner& right) {
                return std::tie(
                    left.masterMeasureIndex,
                    left.position,
                    left.initial)
                    < std::tie(
                    right.masterMeasureIndex,
                    right.position,
                    right.initial);
            });
        return true;
    } catch (...) {
        owners->clear();
        errOut = "official tempo owner enumeration threw";
        return false;
    }
}

static bool fileLooksWritten(const std::string& path) {
    QFileInfo info(QString::fromUtf8(path.c_str()));
    return info.exists() && info.size() > 1024;
}

static void renderPreparedContext(void* context) {
    DirectExportState* state = g_directExportState;
    if (!state) {
        logMsg("[DIRECT] render requested without thread-local export state");
        return;
    }

    int pageCount = 0;
    if (context) {
        auto* words = reinterpret_cast<uintptr_t*>(context);
        pageCount = static_cast<int>((words[40] - words[39]) / 208);
    }
    auto exportUsingQPdfWriter = reinterpret_cast<void (__fastcall *)(
        void*, int, int, const QString*, int)>(
            guitarProMainVa(0x7FF68EF25D00));

    logMsg("[DIRECT] prepared context=%p pages=%d worker=%p output=%s",
           context,
           pageCount,
           reinterpret_cast<void*>(exportUsingQPdfWriter),
           state->outputPath.c_str());

    const bool canRender = context && pageCount > 0 && exportUsingQPdfWriter
        && !state->outputPath.empty();
    if (!canRender) return;

    const bool dumpLayout = !state->layoutPath.empty();

    QString qOutput = QString::fromUtf8(state->outputPath.c_str());
    QDir().mkpath(QFileInfo(qOutput).absolutePath());
    QFile::remove(qOutput);

    bool captureLayout = false;
    if (dumpLayout) {
        g_layoutDumpActive = true;
        g_layoutSystems.clear();
        g_layoutMeasureViews.clear();
        g_timeSignatures.clear();
        g_keySignatures.clear();
        g_restViews.clear();
        g_noteGlyphs.clear();
        g_lyricEvents.clear();
        g_tempoViewCaptures.clear();
        g_tempoIndications.clear();
        g_paintDeviceTranslations.clear();
        g_renderedSystemFirstMeasureIndex = -1;
        g_tempoBandElementRenderDepth = 0;
        if (!installNativeLayoutHooks()) {
            logMsg("[LAYOUT] Failed to install native hooks");
            g_layoutDumpActive = false;
        } else {
            if (!g_restViewHookReady.load()) {
                markRestViewsUnresolved("RestView hook is unavailable");
            }
            if (!g_noteGeometryHookReady.load()) {
                noteGeometryError("native note geometry hook is unavailable");
            }
            if (!g_lyricBandElementHookReady.load()) {
                markLyricEventsUnresolved("lyric View hook is unavailable");
            }
            if (!g_tempoIndicationHooksReady.load()) {
                markTempoIndicationsUnresolved(
                    "TempoIndication hooks are unavailable");
            }
            collectLayoutPages(context);
            captureLayout = true;
        }
    }

    exportUsingQPdfWriter(context, 0, pageCount, &qOutput, state->dpi);
    if (captureLayout) {
        const bool matrixStackUnbalanced = std::any_of(
            g_paintDeviceTranslations.begin(),
            g_paintDeviceTranslations.end(),
            [](const auto& entry) { return !entry.second.empty(); });
        if (matrixStackUnbalanced || g_tempoBandElementRenderDepth != 0) {
            noteGeometryError("render matrix/depth state is unbalanced");
            markTempoIndicationsUnresolved(
                "TempoIndication render matrix/depth state is unbalanced");
        }
        g_layoutDumpActive = false;
        g_paintDeviceTranslations.clear();
        g_renderedSystemFirstMeasureIndex = -1;
        g_tempoBandElementRenderDepth = 0;
    }

    state->pdfOk = fileLooksWritten(state->outputPath);
    state->layoutOk = !dumpLayout;
    if (dumpLayout) {
        state->layoutOk = captureLayout
            && state->layoutError.empty()
            && writeNativeLayoutDump(
                state->layoutPath, state->pdfOk, state->sourceTrackIndex);
    }
    logMsg(
        "[DIRECT] prepared export returned pdf_ok=%d layout_ok=%d",
        state->pdfOk ? 1 : 0,
        state->layoutOk ? 1 : 0);
}

static bool directExportPreparedContext(
        const std::shared_ptr<void>& context,
        const gp::core::ScoreView& view,
        const gp::core::style::Stylesheet& stylesheet,
        const gp::core::Track& selectedTrack,
        const gp::core::MasterTrack& masterTrack,
        const std::string& outputPath,
        const std::string& layoutPath,
        std::string& errOut,
        int sourceTrackIndex,
        bool captureNoteGeometry) {
    if (!context) {
        errOut = "prepared engraving context is empty";
        return false;
    }

    DirectExportState state;
    state.outputPath = outputPath;
    state.layoutPath = layoutPath;
    state.stylesheet = &stylesheet;
    state.sourceTrackIndex = sourceTrackIndex;
    state.captureNoteGeometry = captureNoteGeometry;
    if (!captureNoteGeometry) state.noteGeometryErrors.push_back("capture disabled for render-equivalence control");
    state.scoreViewMultiRest = view.isMultiRest();
    if (!layoutPath.empty()
            && !collectOriginalMasterBarRanges(
                view, &state.masterBarRanges, errOut)) {
        return false;
    }
    if (!layoutPath.empty()
            && !collectTempoOwners(masterTrack, &state.tempoOwners, errOut)) {
        return false;
    }
    g_directExportState = &state;
    if (!layoutPath.empty()) {
        collectSelectedTrackLocations(selectedTrack);
    }
    renderPreparedContext(context.get());
    g_directExportState = nullptr;
    if (!state.pdfOk) {
        errOut = "prepared-context PDF worker did not create output";
        return false;
    }
    if (!state.layoutOk) {
        errOut = state.layoutError.empty()
            ? "prepared-context layout worker did not create output"
            : state.layoutError;
        return false;
    }
    return true;
}

struct PreparedCoreExport {
    std::unique_ptr<gp::core::style::Stylesheet> stylesheet;
    std::unique_ptr<gp::core::ScoreView> view;
    std::shared_ptr<void> context;
};

static bool prepareCoreScoreViewForExport(
        gp::core::Score& score,
        PreparedCoreExport* preparedOut,
        std::string& errOut) {
    using fn_CreateEngravingContext = std::shared_ptr<void>* (__fastcall *)(
        std::shared_ptr<void>*);
    using fn_SetContextView = void (__fastcall *)(void*, gp::core::ScoreView*);
    using fn_SetContextInt = void (__fastcall *)(void*, int);
    using fn_SetContextStylesheet = void (__fastcall *)(void*, void*);
    using fn_UpdateContext = void (__fastcall *)(void*, unsigned char);
    using fn_UpdateAdaptiveLayouts = void (__fastcall *)(void*);
    using fn_RebuildMasterBars = void (__fastcall *)(void*, void*);

    auto createContext = reinterpret_cast<fn_CreateEngravingContext>(
        guitarProMainVa(0x7FF68F284CA0));
    auto setView = reinterpret_cast<fn_SetContextView>(
        guitarProMainVa(0x7FF68F290800));
    auto setOption0 = reinterpret_cast<fn_SetContextInt>(
        guitarProMainVa(0x7FF68F2907B0));
    auto setLayout = reinterpret_cast<fn_SetContextInt>(
        guitarProMainVa(0x7FF68F290770));
    auto setStylesheet = reinterpret_cast<fn_SetContextStylesheet>(
        guitarProMainVa(0x7FF68F290DA0));
    auto setOption1 = reinterpret_cast<fn_SetContextInt>(
        guitarProMainVa(0x7FF68F2904E0));
    auto update = reinterpret_cast<fn_UpdateContext>(
        guitarProMainVa(0x7FF68F296180));
    auto updateAdaptiveLayouts = reinterpret_cast<fn_UpdateAdaptiveLayouts>(
        guitarProMainVa(0x7FF68F28D330));
    auto rebuildMasterBars = reinterpret_cast<fn_RebuildMasterBars>(
        guitarProMainVa(0x7FF68F29B0C0));
    if (!createContext || !setView || !setOption0 || !setLayout
            || !setStylesheet || !setOption1 || !update
            || !updateAdaptiveLayouts || !rebuildMasterBars) {
        errOut = "Guitar Pro engraving context functions are unavailable";
        return false;
    }
    if (!g_pageDecorationHook.installed || !preparedOut) {
        errOut = "Guitar Pro page decoration hook is unavailable";
        return false;
    }

    auto& activeView = score.activeView();
    void* stylesheet = *reinterpret_cast<void**>(
        reinterpret_cast<unsigned char*>(&activeView) + 152);
    if (!stylesheet) {
        errOut = "Guitar Pro active view has no stylesheet";
        return false;
    }

    std::shared_ptr<void> context;
    createContext(&context);
    if (!context) {
        errOut = "Guitar Pro failed to create an engraving context";
        return false;
    }

    const size_t systemsBefore = activeView.systems().size();
    setView(context.get(), &activeView);
    setOption0(context.get(), 0);
    setLayout(context.get(), 2);
    setStylesheet(context.get(), stylesheet);
    setOption1(context.get(), 0);
    rebuildMasterBars(context.get(), nullptr);
    update(context.get(), 1);
    const std::vector<int> emptyLayouts;
    gp::core::LayoutHandler::setLayouts(score, emptyLayouts);
    score.activeView().setMultiRest(false);
    rebuildMasterBars(context.get(), nullptr);
    *reinterpret_cast<int*>(
        static_cast<unsigned char*>(context.get()) + 652) = 2;
    updateAdaptiveLayouts(context.get());
    context.reset();

    PreparedCoreExport prepared;
    const std::shared_ptr<gp::core::style::Stylesheet> emptyStylesheet;
    prepared.stylesheet = std::make_unique<gp::core::style::Stylesheet>(
        emptyStylesheet);
    prepared.stylesheet->applyModel(score.newStylesheet());
    gp::core::style::Stylesheet::setupStyleForExport(*prepared.stylesheet);
    const std::optional<
        gp::core::style::generated::BarNumberLayout::Frequency
    > hiddenBarNumbers(
        gp::core::style::generated::BarNumberLayout::Frequency::None);
    gp::core::style::setBarNumberLayoutFrequency(
        *prepared.stylesheet,
        hiddenBarNumbers);
    const std::optional<am::painting::Color> transparentBackground(
        am::painting::Color::Transparent);
    gp::core::style::setPageLayoutBackgroundColor(
        *prepared.stylesheet,
        transparentBackground);

    prepared.view = std::make_unique<gp::core::ScoreView>();
    prepared.view->applyModel(activeView);
    prepared.view->setMultiRest(false);
    createContext(&prepared.context);
    if (!prepared.context) {
        errOut = "Guitar Pro failed to create the export engraving context";
        return false;
    }
    setView(prepared.context.get(), prepared.view.get());
    setOption0(prepared.context.get(), 0);
    setLayout(prepared.context.get(), 2);
    setStylesheet(prepared.context.get(), prepared.stylesheet.get());
    setOption1(prepared.context.get(), 0);
    rebuildMasterBars(prepared.context.get(), nullptr);
    update(prepared.context.get(), 1);
    *preparedOut = std::move(prepared);
    logMsg("[CORE-DIRECT] Prelayout systems=%zu->%zu",
           systemsBefore,
           activeView.systems().size());
    return true;
}

static bool initializeCoreTemplate(const QFileInfo& inputInfo,
                                   std::shared_ptr<gp::core::Score>& score,
                                   std::string& errOut) {
    using fn_CreateTemplateScore = std::shared_ptr<gp::core::Score>* (__fastcall *)(
        std::shared_ptr<gp::core::Score>*, const QString*);
    auto createTemplateScore = reinterpret_cast<fn_CreateTemplateScore>(
        guitarProMainVa(0x7FF68ECC61B0));
    if (!createTemplateScore) {
        errOut = "Guitar Pro template score initializer is unavailable";
        return false;
    }
    const QString fileName = inputInfo.fileName();
    createTemplateScore(&score, &fileName);
    if (!score) {
        errOut = "Guitar Pro template score initializer returned no score";
        return false;
    }
    return true;
}

static std::shared_ptr<gp::core::Score> loadCoreScoreWithoutAppDocument(
        const std::string& inputPath,
        std::string& errOut,
        std::string& errorCodeOut) {
    errorCodeOut.clear();
    const QString qInput = QString::fromUtf8(inputPath.c_str());
    const QFileInfo inputInfo(qInput);
    if (!inputInfo.isFile()) {
        errOut = "input file does not exist: " + inputPath;
        return nullptr;
    }

    const QString suffix = inputInfo.suffix().toLower();
    if (suffix != QStringLiteral("gp")
            && suffix != QStringLiteral("gpx")
            && suffix != QStringLiteral("gp3")
            && suffix != QStringLiteral("gp4")
            && suffix != QStringLiteral("gp5")) {
        errOut = "unsupported Guitar Pro input extension: "
            + suffix.toStdString();
        return nullptr;
    }

    std::shared_ptr<gp::core::Score> score;
    try {
        initializeCoreTemplate(inputInfo, score, errOut);
        if (!score) {
            return nullptr;
        }

        const QString fileName = inputInfo.fileName();

        QFile sourceFile(qInput);
        if (!sourceFile.open(QIODevice::ReadOnly)) {
            errOut = "failed to read input through Qt: " + inputPath;
            return nullptr;
        }
        const QByteArray sourceBytes = sourceFile.readAll();
        sourceFile.close();

        am::filesystem::RAMFileSystem fileSystem;
        auto stagedFile = fileSystem.openHandle(
            fileName,
            am::filesystem::FileSystem::Mode::Write);
        if (!stagedFile
                || stagedFile->write(sourceBytes.constData(), sourceBytes.size())
                    != sourceBytes.size()) {
            errOut = "failed to stage input in Guitar Pro memory file system: " + inputPath;
            return nullptr;
        }
        stagedFile.reset();

        auto file = fileSystem.openHandle(
            fileName,
            am::filesystem::FileSystem::Mode::Read);
        if (!file) {
            errOut = "failed to open input through Guitar Pro file system: " + inputPath;
            return nullptr;
        }

        QString importerExtension = suffix;
        auto* importer = gp::core::Core::instance().importerByHandleAndExtension(
            *file, importerExtension);
        if (!importer) {
            errOut = "Guitar Pro found no importer for: " + inputPath;
            return nullptr;
        }

        try {
            score->load(*file, importer);
        } catch (...) {
            errorCodeOut = "official_primary_load_rejected";
            errOut = "Guitar Pro importer rejected source during primary load: "
                + inputPath;
            return nullptr;
        }
    } catch (...) {
        errOut = "Guitar Pro core loader threw while reading: " + inputPath;
        return nullptr;
    }
    logMsg("[CORE-DIRECT] Parsed %s score=%p", inputPath.c_str(), score.get());
    return score;
}

static std::shared_ptr<gp::core::Score> scoreForSessionInput(
        const std::string& inputPath,
        std::string& errOut,
        std::string& errorCodeOut) {
    const QFileInfo inputInfo(QString::fromUtf8(inputPath.c_str()));
    const QString canonicalPath = inputInfo.canonicalFilePath();
    if (canonicalPath.isEmpty()) {
        errOut = "input file does not exist: " + inputPath;
        return nullptr;
    }
    const std::string canonical = canonicalPath.toUtf8().toStdString();
    if (g_sessionScore) {
        if (canonical != g_sessionScorePath) {
            errorCodeOut = "session_source_mismatch";
            errOut = "Guitar Pro export session is already bound to a different source";
            return nullptr;
        }
        return g_sessionScore;
    }

    auto score = loadCoreScoreWithoutAppDocument(
        canonical, errOut, errorCodeOut);
    if (!score) return nullptr;
    g_sessionScorePath = canonical;
    g_sessionScore = score;
    logMsg("[CORE-DIRECT] Session bound to %s", canonical.c_str());
    return score;
}

static ConvertResult releaseSessionSource() {
    std::lock_guard<std::mutex> lock(g_directConvertMutex);
    ConvertResult result;

    runOnMainThread([&]() {
        if (g_directExportState != nullptr || g_layoutDumpActive) {
            result.errorCode = "export_active";
            result.error = "cannot release source while an export is active";
            return;
        }

        g_sessionScore.reset();
        g_sessionScorePath.clear();
        g_layoutPages.clear();
        g_layoutSystems.clear();
        g_timeSignatures.clear();
        g_keySignatures.clear();
        g_layoutMeasureViews.clear();
        g_restViews.clear();
        g_lyricEvents.clear();
        g_tempoViewCaptures.clear();
        g_tempoIndications.clear();
        g_paintDeviceTranslations.clear();
        g_renderedSystemFirstMeasureIndex = -1;
        g_tempoBandElementRenderDepth = 0;
        g_layoutDumpActive = false;
        result.ok = true;
        logMsg("[CORE-DIRECT] Session source released");
    });
    return result;
}

static bool prepareLoadedCoreScoreForExport(
        gp::core::Score& score,
        PreparedCoreExport* preparedOut,
        std::string& errOut) {
    try {
        // A measure sample must have one official BarView per semantic bar.
        // Saved .gp views may collapse consecutive empty bars into one multi-rest.
        logMsg(
            "[CORE-DIRECT] Source ScoreView multi-rest=%d",
            score.activeView().isMultiRest() ? 1 : 0);
        const std::vector<int> emptyLayouts;
        gp::core::LayoutHandler::setLayouts(score, emptyLayouts);
        auto& exportView = score.activeView();
        exportView.setMultiRest(false);
        exportView.clearMasterBars();
        exportView.clearSystems();
        logMsg(
            "[CORE-DIRECT] Export ScoreView multi-rest=%d master-bars=%zu",
            exportView.isMultiRest() ? 1 : 0,
            exportView.masterBars().size());
        if (!prepareCoreScoreViewForExport(score, preparedOut, errOut)) {
            return false;
        }
    } catch (...) {
        errOut = "Guitar Pro core layout preparation threw";
        return false;
    }
    return true;
}

static const char* supportedInstrumentKind(
        gp::core::InstrumentSet::Type type) {
    if (gp::core::InstrumentSet::isBass(type)) return "bass";
    if (gp::core::InstrumentSet::isGuitar(type)) return "guitar";
    return nullptr;
}

static TrackListResult doCoreTrackList(const std::string& inputPath) {
    std::lock_guard<std::mutex> lock(g_directConvertMutex);
    TrackListResult result;

    runOnMainThread([&]() {
        if (!waitForDirectReadyOnMainThread(30000)) {
            result.error = "direct core is not ready";
            return;
        }

        std::string loadError;
        std::string loadErrorCode;
        const auto score = scoreForSessionInput(
            inputPath, loadError, loadErrorCode);
        if (!score) {
            result.errorCode = loadErrorCode;
            result.error = loadError;
            return;
        }

        try {
            result.tracks.reserve(score->trackCount());
            for (unsigned int sourceIndex = 0;
                 sourceIndex < score->trackCount();
                 ++sourceIndex) {
                const auto track = score->track(sourceIndex);
                if (!track) {
                    result.error = "Guitar Pro returned an empty source track";
                    result.tracks.clear();
                    return;
                }
                const auto type = track->type();
                const char* kind = supportedInstrumentKind(type);
                if (!kind) continue;
                result.tracks.push_back({
                    static_cast<int>(sourceIndex),
                    track->name(),
                    kind,
                });
            }
        } catch (...) {
            result.error = "Guitar Pro track enumeration threw";
            result.tracks.clear();
            return;
        }
        result.ok = true;
    });
    return result;
}

static bool configureCoreScoreForTrackExport(
        gp::core::Score& score,
        int selectedTrackIndex,
        std::string& errOut) {
    if (selectedTrackIndex < 0
            || selectedTrackIndex >= static_cast<int>(score.trackCount())) {
        errOut = "requested source track index is out of range";
        return false;
    }
    auto& view = score.activeView();
    const auto selectedTrack = score.track(
        static_cast<unsigned int>(selectedTrackIndex));
    if (!selectedTrack) {
        errOut = "requested source track is empty";
        return false;
    }
    if (selectedTrack->staffCount() == 0) {
        errOut = "requested source track has no staff";
        return false;
    }
    for (unsigned int staffIndex = 0;
         staffIndex < selectedTrack->staffCount();
         ++staffIndex) {
        const auto staff = selectedTrack->staff(staffIndex);
        if (!staff || staff->index() != staffIndex) {
            errOut = "requested source track has an invalid staff identity";
            return false;
        }
    }
    for (unsigned int trackIndex = score.trackCount(); trackIndex > 0;) {
        --trackIndex;
        view.removeTrackViewGroupAtTrackIndex(trackIndex);
    }
    view.insertTrackViewGroup(
        static_cast<unsigned int>(selectedTrackIndex),
        selectedTrack.get(),
        gp::core::TrackView::Type::Tablature);
    const unsigned int groupCount = view.trackViewGroupCount();
    if (groupCount != 1) {
        errOut = "Guitar Pro did not create exactly one selected track view group";
        return false;
    }
    auto& group = view.trackViewGroup(0);
    group.setVisible(true);
    group.setStandardNotation(false);
    group.setSlash(false);
    group.setNumberedNotation(false);
    group.setTablature(true);
    const auto groupTrack = group.track();
    if (!groupTrack
            || groupTrack.get() != selectedTrack.get()
            || group.trackIndex()
                != static_cast<unsigned int>(selectedTrackIndex)) {
        errOut = "Guitar Pro track view group does not bind the selected track";
        return false;
    }
    if (!group.isVisible()
            || group.hasStandardNotation()
            || group.hasSlash()
            || group.hasNumberedNotation()
            || !group.hasTablature()) {
        errOut = "Guitar Pro track view group is not visible TAB-only";
        return false;
    }
    return true;
}

static std::shared_ptr<gp::core::Score> loadScoreWithoutAppDocument(
        const std::string& inputPath,
        std::string& errOut,
        std::string& errorCodeOut,
        PreparedCoreExport* preparedOut,
        int selectedTrackIndex) {
    auto score = scoreForSessionInput(
        inputPath, errOut, errorCodeOut);
    if (!score) return nullptr;
    try {
        score->views().clear();
        gp::core::io::gp_Score_InitScoreViews(*score);
    } catch (...) {
        errOut = "Guitar Pro score-view reset threw";
        return nullptr;
    }
    if (!configureCoreScoreForTrackExport(
            *score, selectedTrackIndex, errOut)) {
        return nullptr;
    }
    if (!prepareLoadedCoreScoreForExport(
            *score,
            preparedOut,
            errOut)) {
        return nullptr;
    }
    return score;
}

static ConvertResult doCoreDirectConvert(const std::string& inputPath,
                                         const std::string& outputPath,
                                         const std::string& layoutPath,
                                         const std::string& officialScorePath,
                                         int selectedTrackIndex,
                                         bool captureNoteGeometry) {
    std::lock_guard<std::mutex> lock(g_directConvertMutex);
    ConvertResult cr;
    std::string error;
    std::string errorCode;

    runOnMainThread([&]() {
        ScopedPageDecorationForce forcePageDecorations;
        if (!waitForDirectReadyOnMainThread(30000)) {
            error = "direct core is not ready";
            return;
        }

        QString qOutput = QString::fromUtf8(outputPath.c_str());
        QDir().mkpath(QFileInfo(qOutput).absolutePath());
        QFile::remove(qOutput);

        if (!layoutPath.empty()) {
            if (!installNativeLayoutHooks()) {
                error = "failed to install native layout hooks before engraving";
                return;
            }
        }

        std::shared_ptr<gp::core::Score> score;
        PreparedCoreExport prepared;
        score = loadScoreWithoutAppDocument(
            inputPath,
            error,
            errorCode,
            &prepared,
            selectedTrackIndex);
        if (!score) return;

        const auto selectedTrack = score->track(
            static_cast<unsigned int>(selectedTrackIndex));
        if (!selectedTrack) {
            error = "selected source track is unavailable after layout preparation";
            return;
        }
        const auto masterTrack = score->masterTrack();
        if (!masterTrack) {
            error = "master track is unavailable after layout preparation";
            return;
        }
        if (!gpomr::writeScoreDump(
                *score,
                *prepared.stylesheet,
                officialScorePath,
                error,
                selectedTrackIndex)) {
            return;
        }
        if (!directExportPreparedContext(
                prepared.context,
                *prepared.view,
                *prepared.stylesheet,
                *selectedTrack,
                *masterTrack,
                outputPath,
                layoutPath,
                error,
                selectedTrackIndex,
                captureNoteGeometry)) {
            return;
        }
        cr.ok = true;
    });

    if (!cr.ok) {
        cr.errorCode = errorCode;
        cr.error = error.empty() ? "core direct convert failed" : error;
        logMsg("[CORE-DIRECT] Failed: %s", cr.error.c_str());
    } else {
        logMsg("[CORE-DIRECT] Success");
    }
    return cr;
}

// ── Pipe Server ──────────────────────────────────────────────────

static HANDLE g_pipeThread = NULL;
static volatile bool g_running = true;
static char g_pipeName[256] = "\\\\.\\pipe\\gpomr_export";

static bool parseTrackIndex(const std::string& text, int* value) {
    if (!value || text.empty()) return false;
    int parsed = 0;
    for (char digit : text) {
        if (digit < '0' || digit > '9') return false;
        const int next = digit - '0';
        if (parsed > ((std::numeric_limits<int>::max)() - next) / 10) {
            return false;
        }
        parsed = parsed * 10 + next;
    }
    *value = parsed;
    return true;
}

static std::string trackListResponse(const TrackListResult& result) {
    if (!result.ok) {
        if (!result.errorCode.empty()) {
            return json::make({
                {"ok", "false"},
                {"error_code", result.errorCode},
                {"error", result.error.empty()
                    ? "Guitar Pro track enumeration failed"
                    : result.error},
            });
        }
        return json::make({
            {"ok", "false"},
            {"error", result.error.empty()
                ? "Guitar Pro track enumeration failed"
                : result.error},
        });
    }

    std::ostringstream response;
    response << "{\"ok\":true,\"tracks\":[";
    for (size_t index = 0; index < result.tracks.size(); ++index) {
        if (index) response << ',';
        const auto& track = result.tracks[index];
        response << "{\"source_track_index\":" << track.sourceTrackIndex
            << ",\"name\":\"" << jsonEscape(track.name) << "\""
            << ",\"instrument_kind\":\""
            << track.instrumentKind << "\"}";
    }
    response << "]}\n";
    return response.str();
}

static DWORD WINAPI pipeServerThread(LPVOID) {
    const char* pipeName = g_pipeName;
    char buf[65536];

    while (g_running) {
        HANDLE pipe = CreateNamedPipeA(
            pipeName,
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1, 65536, 65536, 0, NULL);

        if (pipe == INVALID_HANDLE_VALUE) {
            Sleep(1000);
            continue;
        }

        if (!ConnectNamedPipe(pipe, NULL) && GetLastError() != ERROR_PIPE_CONNECTED) {
            CloseHandle(pipe);
            continue;
        }

        while (g_running) {
            DWORD bytesRead = 0;
            if (!ReadFile(pipe, buf, sizeof(buf) - 1, &bytesRead, NULL) || bytesRead == 0)
                break;
            buf[bytesRead] = '\0';

            std::string input(buf, bytesRead);
            std::string::size_type lineStart = 0;

            while (lineStart < input.size()) {
                auto lineEnd = input.find('\n', lineStart);
                if (lineEnd == std::string::npos) lineEnd = input.size();

                std::string line = input.substr(lineStart, lineEnd - lineStart);
                lineStart = lineEnd + 1;

                if (line.empty() || line[0] != '{') continue;

                std::string cmd = json::get(line, "cmd");
                std::string response;

                if (cmd == "quit") {
                    response = json::make({{"ok", "true"}});
                    DWORD written;
                    WriteFile(pipe, response.c_str(), (DWORD)response.size(), &written, NULL);
                    FlushFileBuffers(pipe);
                    g_running = false;
                    break;
                } else if (cmd == "release_source") {
                    const auto released = releaseSessionSource();
                    if (released.ok) {
                        response = json::make({{"ok", "true"}});
                    } else if (!released.errorCode.empty()) {
                        response = json::make({
                            {"ok", "false"},
                            {"error_code", released.errorCode},
                            {"error", released.error},
                        });
                    } else {
                        response = json::make({
                            {"ok", "false"},
                            {"error", released.error.empty()
                                ? "failed to release source"
                                : released.error},
                        });
                    }
                } else if (cmd == "list_tracks") {
                    const std::string inPath = json::get(line, "input");
                    if (inPath.empty()) {
                        response = json::make({
                            {"ok", "false"},
                            {"error", "missing list_tracks input"},
                        });
                    } else {
                        response = trackListResponse(doCoreTrackList(inPath));
                    }
                } else if (cmd == "export") {
                    std::string inPath = json::get(line, "input");
                    std::string outPath = json::get(line, "output");
                    std::string layoutPath = json::get(line, "layout_output");
                    std::string officialScorePath = json::get(
                        line, "official_score_output");
                    const std::string trackIndexText = json::get(line, "track_index");
                    const std::string tabOnlyText = json::get(line, "tab_only");
                    const std::string geometryText = json::get(line, "capture_note_geometry");
                    int selectedTrackIndex = -1;
                    const bool validTrackIndex = parseTrackIndex(
                        trackIndexText, &selectedTrackIndex);

                    if (inPath.empty()
                            || outPath.empty()
                            || layoutPath.empty()
                            || officialScorePath.empty()
                            || !validTrackIndex
                            || tabOnlyText != "true"
                            || (!geometryText.empty() && geometryText != "true" && geometryText != "false")) {
                        response = json::make({
                            {"ok", "false"},
                            {"error", "missing or invalid export fields"}
                        });
                    } else {
                        auto cr = doCoreDirectConvert(
                            inPath,
                            outPath,
                            layoutPath,
                            officialScorePath,
                            selectedTrackIndex,
                            geometryText != "false");
                        if (cr.ok) {
                            response = json::make({{"ok", "true"}});
                        } else if (!cr.errorCode.empty()) {
                            response = json::make({
                                {"ok", "false"},
                                {"error_code", cr.errorCode},
                                {"error", cr.error},
                            });
                        } else {
                            response = json::make({{"ok", "false"}, {"error", cr.error}});
                        }
                    }
                } else {
                    response = json::make({{"ok", "false"}, {"error", "unknown cmd: " + cmd}});
                }

                DWORD written;
                WriteFile(pipe, response.c_str(), (DWORD)response.size(), &written, NULL);
                FlushFileBuffers(pipe);
            }
        }

        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
    }

    return 0;
}


// ── DLL Entry ────────────────────────────────────────────────────

static void logModuleBases() {
    const char* dlls[] = {"GPCore.dll", "AMPainting.dll", "AMUtils.dll",
                          "Qt5Core.dll", "Qt5Gui.dll", "Qt5Widgets.dll",
                          "GuitarPro.exe"};
    for (auto name : dlls) {
        HMODULE h = GetModuleHandleA(name);
        if (h) logMsg("[MODULE] %s -> base=%p", name, (void*)h);
    }
}

static LONG WINAPI crashHandler(EXCEPTION_POINTERS* ep) {
    logMsg("[CRASH] Exception 0x%08lx at %p",
           ep->ExceptionRecord->ExceptionCode,
           ep->ExceptionRecord->ExceptionAddress);
    if (ep && ep->ContextRecord) {
        auto* ctx = ep->ContextRecord;
        logMsg("[CRASH] regs rip=%p rsp=%p rbp=%p rax=%p rbx=%p rcx=%p rdx=%p rsi=%p rdi=%p r8=%p r9=%p",
               (void*)ctx->Rip,
               (void*)ctx->Rsp,
               (void*)ctx->Rbp,
               (void*)ctx->Rax,
               (void*)ctx->Rbx,
               (void*)ctx->Rcx,
               (void*)ctx->Rdx,
               (void*)ctx->Rsi,
               (void*)ctx->Rdi,
               (void*)ctx->R8,
               (void*)ctx->R9);
        auto* stack = reinterpret_cast<uintptr_t*>(ctx->Rsp);
        for (int i = 0; i < 12; ++i) {
            logMsg("[CRASH] stack[%02d]=%p", i, (void*)stack[i]);
        }
    }

    // 记录崩溃所在模块。
    HMODULE hMod = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                       (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &hMod);
    if (hMod) {
        char modName[MAX_PATH];
        GetModuleFileNameA(hMod, modName, MAX_PATH);
        ULONG_PTR offset = (ULONG_PTR)ep->ExceptionRecord->ExceptionAddress - (ULONG_PTR)hMod;
        logMsg("[CRASH] In module: %s + 0x%llx", modName, (unsigned long long)offset);
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static DWORD WINAPI initializeBridgeThread(LPVOID) {
    SetUnhandledExceptionFilter(crashHandler);

    logMsg("========================================");
    char envBuf[256];
    if (GetEnvironmentVariableA("GPOMR_EXPORT_PIPE", envBuf, sizeof(envBuf)) > 0) {
        strncpy(g_pipeName, envBuf, sizeof(g_pipeName) - 1);
    }
    if (GetEnvironmentVariableA("GPOMR_EXPORT_READY_EVENT", envBuf, sizeof(envBuf)) > 0) {
        g_directReadyEvent = CreateEventA(NULL, TRUE, FALSE, envBuf);
        logMsg("[INIT] Ready event: %s handle=%p", envBuf, g_directReadyEvent);
    }
    logMsg("[INIT] gpomr_native_export.dll loaded");
    logMsg("[INIT] Pipe name: %s", g_pipeName);
    logModuleBases();

    installPageDecorationHook();

    g_running = true;
    startDirectReadyProbe();
    g_pipeThread = CreateThread(NULL, 0, pipeServerThread, NULL, 0, NULL);
    logMsg("[INIT] Pipe server started");
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
        HANDLE initThread = CreateThread(
            NULL, 0, initializeBridgeThread, NULL, 0, NULL);
        if (initThread) {
            CloseHandle(initThread);
        }
    } else if (reason == DLL_PROCESS_DETACH) {
        logMsg("[SHUTDOWN] DLL unloading");
        g_running = false;
        removeNativeLayoutHooks();
        removeInlineHook(&g_pageDecorationHook);
        if (g_pipeThread) {
            WaitForSingleObject(g_pipeThread, 3000);
            CloseHandle(g_pipeThread);
        }
        if (g_directReadyEvent) {
            CloseHandle(g_directReadyEvent);
            g_directReadyEvent = NULL;
        }
        if (g_logFile) fclose(g_logFile);
    }
    return TRUE;
}
