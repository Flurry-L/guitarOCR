#pragma once
/**
 * gp_stubs.h
 * Guitar Pro DLL ABI declarations.
 *
 * 命名空间映射（undname.exe 和 dumpbin 核对）：
 *   gp::core::Score               -> GPCore.dll
 *   gp::core::ScoreView           -> GPCore.dll
 *   gp::core::LayoutHandler       -> GPCore.dll
 *   gp::core::view::SystemView    -> GPCore.dll
 *   am::filesystem::FileHandle    -> AMUtils.dll
 *   am::filesystem::FileSystem    -> AMUtils.dll
 *   am::filesystem::RealFileSystem -> AMUtils.dll
 *
 * MSVC 名字修饰要点：
 *   - "class" 在修饰名中对应 V 前缀，"struct" 对应 U 前缀。
 *   - "virtual" 方法对应 UEAA，非 virtual 方法对应 QEAA。
 *   下面的声明必须和 DLL ABI 严格一致。
 */

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>
#include <QString>

namespace am { namespace music {
enum class Accidental : int;
enum class Dynamic : int;
}} // am::music

namespace am { namespace utils {
class rational {
public:
    int numerator() const;
    int denominator() const;

private:
    alignas(8) unsigned char _opaque[32];
};
static_assert(sizeof(rational) == 32, "rational ABI size mismatch");
}} // am::utils

namespace am { namespace thread {

class ThreadPool {
public:
    static ThreadPool& instance();
    void resize(size_t threadCount);
};

}} // am::thread

// ──────────────────────────────────────────────────────────────────
// am::painting types used by View ABI declarations and export styling.
// Guitar Pro 把这些类型声明为 class，因此这里也必须使用 class。
// struct 与 class 会影响 MSVC 修饰名前缀：U 与 V。
// ──────────────────────────────────────────────────────────────────
namespace am { namespace painting {

#pragma pack(push, 8)
class Point  { public: double x = 0, y = 0; };
class Rect   { public: double x = 0, y = 0, w = 0, h = 0; };
class IElement {
public:
    Rect BBox() const;
};
class Color  {
public:
    float r = 0, g = 0, b = 0, a = 1;
    static Color Transparent;
};
class FormattedText {
public:
    virtual ~FormattedText();
    const std::string& text() const;

private:
    alignas(8) unsigned char _opaque[24];
};
#pragma pack(pop)

static_assert(sizeof(FormattedText) == 32, "FormattedText ABI size mismatch");

}} // am::painting


// ──────────────────────────────────────────────────────────────────
// am::filesystem::FileHandle（AMUtils.dll）
// 不能直接构造，只能通过 FileSystem::openHandle() 获取。
// ──────────────────────────────────────────────────────────────────
namespace am { namespace filesystem {

class FileHandle {
public:
    void close();
    long read(void* buf, long size);
    long write(const void* buf, long size);
    std::string readAll();
    QString filename() const;
    int   readInt();
    short readShort();
    char  readByte();
    void setLittleEndian();
    void setBigEndian();
    long  seek(long offset, int origin);
    long  tell();
    unsigned size();

private:
    FileHandle() = delete;
    FileHandle(const FileHandle&) = delete;
    alignas(8) unsigned char _opaque[256];
};


// am::filesystem::FileSystem 基类（AMUtils.dll）
// FileSystem 是多态类型：??1FileSystem@filesystem@am@@UEAA@XZ 导出了虚析构。
// 必须声明 virtual ~FileSystem()，让编译器把 vptr 放在 FileSystem 偏移 0，
// 从而匹配 DLL 内部布局。这里使用内联空实现。
class FileSystem {
public:
    // 枚举值匹配 Guitar Pro 的 FileSystem::Mode；Read=0 是常规取值。
    enum class Mode : int { Read = 0, Write = 1, ReadWrite = 2 };

    // 虚析构是必需的；DLL 中 FileSystem 本身就是多态类型。
    // 如果缺少它，MSVC 会把 RealFileSystem 的 vptr 放到 FileSystem 子对象之前，
    // 导致成员偏移与 DLL 期望不一致。
    virtual ~FileSystem() {}

    // ?openHandle@FileSystem@filesystem@am@@QEAA?AV?$unique_ptr@VFileHandle@filesystem@am@@
    //   V?$function@$$A6AXPEAVFileHandle@filesystem@am@@@Z@std@@@std@@AEBVQString@@W4Mode@123@@Z
    std::unique_ptr<FileHandle, std::function<void(FileHandle*)>>
        openHandle(const QString& path, Mode mode = Mode::Read);

protected:
    // 512 字节是保守预留；实际 FileSystem 成员数据（不含 vptr）更小。
    // 上面有 virtual ~FileSystem() 后，布局为 [vptr:8][_opaque:512] = 520 字节。
    // DLL 在偏移 0 写 vptr、偏移 8 写成员数据，正好匹配。
    alignas(8) unsigned char _opaque[512];
};


// am::filesystem::RealFileSystem（AMUtils.dll）：磁盘文件系统。
// 构造函数需要传入根路径。
class RealFileSystem : public FileSystem {
public:
    // ??0RealFileSystem@filesystem@am@@QEAA@AEBVQString@@@Z
    explicit RealFileSystem(const QString& rootPath);
    // ??1RealFileSystem@filesystem@am@@UEAA@XZ  (virtual destructor)
    virtual ~RealFileSystem();

private:
    alignas(8) unsigned char _opaque_derived[1024];
};


// am::filesystem::RAMFileSystem（AMUtils.dll）：应用加载器先把本地
// 文件复制到这里，使 importer 只看到文件名而不是本机绝对路径。
class RAMFileSystem : public FileSystem {
public:
    RAMFileSystem();
    virtual ~RAMFileSystem();

private:
    alignas(8) unsigned char _opaque_derived[1024];
};

}} // am::filesystem


namespace gp { namespace core { class Score; }}
namespace gp { namespace core { namespace io {

class Importer;

// ?gp_Score_InitScoreViews@io@core@gp@@YAXAEAVScore@23@@Z
void gp_Score_InitScoreViews(gp::core::Score& score);
// ?gp_Score_InitScoreCursor@io@core@gp@@YAXAEAVScore@23@II@Z
void gp_Score_InitScoreCursor(gp::core::Score& score, unsigned int trackIndex, unsigned int barIndex);
// ?gp_Score_InitAutoSystemLayouts@io@core@gp@@YAXAEAVScore@23@@Z
void gp_Score_InitAutoSystemLayouts(gp::core::Score& score);

}}} // gp::core::io


// ──────────────────────────────────────────────────────────────────
// gp::core::view::SystemView（GPCore.dll）
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core {

    class Beat;
    class Bar;
    class Note;

namespace view {

class SystemAttachedView;
class SystemView;

enum class Visibility : int {
    Visible = 0,
    Hidden = 1,
    Gone = 2,
};

enum class BandElementType : int {
    Tempo = 2,
    Lyrics = 0x29,
};

namespace generated {

class BarViewBase {
public:
    const std::shared_ptr<am::painting::IElement>&
        timeSignatureElement() const;
};

class MasterBarViewBase {
public:
    int firstBarIndex() const;
    int lastBarIndex() const;
};

class SystemViewBase {
public:
    int firstMasterBarIndex() const;
    int masterBarCount() const;
    const std::vector<std::shared_ptr<SystemAttachedView>>&
        systemLabels() const;
};

class SystemAttachedViewBase {
public:
    const std::shared_ptr<am::painting::IElement>& element() const;
};

} // generated

class BarView : public generated::BarViewBase {
public:
    std::shared_ptr<Bar> model() const;
};

class MasterBarView : public generated::MasterBarViewBase {
public:
    bool isMultirest() const;
};

class SystemAttachedView : public generated::SystemAttachedViewBase {
public:
    std::shared_ptr<SystemView> systemView() const;
};

    } // view
}} // gp::core

// ──────────────────────────────────────────────────────────────────
// gp::core::style::Stylesheet（GPCore.dll）
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core { namespace style {

namespace generated { namespace BarNumberLayout {

// Values verified against GPCore's BarNumberLayoutFrequency protobuf enum.
enum class Frequency : int {
    Undefined = 0,
    EachBar = 1,
    FirstSystemBar = 2,
    Interval = 3,
    None = 4,
};

}} // generated::BarNumberLayout

namespace generated { namespace SystemLabelStyle {

enum class Mode : int {
    Full = 0,
    Abbreviated = 1,
    None = 2,
};

}} // generated::SystemLabelStyle

namespace generated { namespace DiagramStyle {

enum class Display : int;

}} // generated::DiagramStyle

namespace generated { namespace BeamStyle {

enum class StemAnchor : int;

}} // generated::BeamStyle

namespace generated { namespace TuningStyle {

enum class Mode : int;
enum class Position : int;

}} // generated::TuningStyle

class Stylesheet {
public:
    explicit Stylesheet(const std::shared_ptr<Stylesheet>& model);
    virtual ~Stylesheet();
    void applyModel(const std::shared_ptr<Stylesheet>& model);
    static void setupStyleForExport(Stylesheet& stylesheet);

private:
    unsigned char _opaque[65528];
};

void setPageLayoutBackgroundColor(
    Stylesheet& stylesheet,
    const std::optional<am::painting::Color>& color);

void setBarNumberLayoutFrequency(
    Stylesheet& stylesheet,
    const std::optional<generated::BarNumberLayout::Frequency>& frequency);

generated::SystemLabelStyle::Mode scoreSystemsFirstSystemLabelModeValue(
    const Stylesheet& stylesheet);
generated::SystemLabelStyle::Mode scoreSystemsNextSystemsLabelModeValue(
    const Stylesheet& stylesheet);
view::Visibility tabBeamVisibilityValue(const Stylesheet& stylesheet);
generated::BeamStyle::StemAnchor tabBeamStemAnchorValue(
    const Stylesheet& stylesheet);
bool tabStaveHideUselessRestsValue(const Stylesheet& stylesheet);
bool tabStaveShowQuarterRestAsDashValue(const Stylesheet& stylesheet);
bool tabStaveAlwaysShowTieNotesValue(const Stylesheet& stylesheet);
bool tabStaveHideTiesBetweenBarValue(const Stylesheet& stylesheet);
bool tabStaveDisplayFretRelativeToCapoValue(const Stylesheet& stylesheet);
generated::TuningStyle::Mode scoreTuningModeValue(
    const Stylesheet& stylesheet);
generated::TuningStyle::Position scoreTuningPositionValue(
    const Stylesheet& stylesheet);
int scoreTuningColumnCountValue(const Stylesheet& stylesheet);
bool scoreTuningBoxedValue(const Stylesheet& stylesheet);
view::Visibility noteFretVisibilityValue(const Stylesheet& stylesheet);
view::Visibility noteFretTrillVisibilityValue(const Stylesheet& stylesheet);
view::Visibility noteFretArtificialHarmonicVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreHeaderDiagramVisibilityValue(
    const Stylesheet& stylesheet);
generated::DiagramStyle::Display scoreHeaderDiagramChordNameDisplayValue(
    const Stylesheet& stylesheet);
generated::DiagramStyle::Display scoreHeaderDiagramFingeringDisplayValue(
    const Stylesheet& stylesheet);
view::Visibility scoreScoreDiagramVisibilityValue(
    const Stylesheet& stylesheet);
generated::DiagramStyle::Display scoreScoreDiagramChordNameDisplayValue(
    const Stylesheet& stylesheet);
generated::DiagramStyle::Display scoreScoreDiagramFingeringDisplayValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderTitleVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderTitleFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderSubtitleVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderSubtitleFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderArtistVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderArtistFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderAlbumVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderAlbumFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderLyricsVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderLyricsFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderMusicVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderMusicFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderWordsAndMusicVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderWordsAndMusicFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageHeaderTabberVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageHeaderTabberFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageHeaderVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageHeaderFieldVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreEvenPageHeaderFieldFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageHeaderVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageHeaderFieldVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreOddPageHeaderFieldFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageFooterVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageFooterCopyrightVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageFooterCopyrightFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageFooterCopyright2VisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageFooterCopyright2FormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreFirstPageFooterPageNumberVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreFirstPageFooterPageNumberFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageFooterVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageFooterCopyrightVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreEvenPageFooterCopyrightFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageFooterCopyright2VisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreEvenPageFooterCopyright2FormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreEvenPageFooterPageNumberVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreEvenPageFooterPageNumberFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageFooterVisibilityValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageFooterCopyrightVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreOddPageFooterCopyrightFormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageFooterCopyright2VisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreOddPageFooterCopyright2FormattedTextValue(
    const Stylesheet& stylesheet);
view::Visibility scoreOddPageFooterPageNumberVisibilityValue(
    const Stylesheet& stylesheet);
am::painting::FormattedText scoreOddPageFooterPageNumberFormattedTextValue(
    const Stylesheet& stylesheet);

}}} // gp::core::style


// ──────────────────────────────────────────────────────────────────
// gp::core::ScoreView（GPCore.dll）
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core {

class Track;

class TrackView {
public:
    enum class Type : int {
        Tablature = 2,
    };
};

class TrackViewGroup {
public:
    bool hasNumberedNotation() const;
    bool hasSlash() const;
    bool hasStandardNotation() const;
    bool hasTablature() const;
    bool isVisible() const;
    std::shared_ptr<Track> track() const;
    unsigned int trackIndex() const;
    void setNumberedNotation(bool enabled);
    void setSlash(bool enabled);
    void setStandardNotation(bool enabled);
    void setTablature(bool enabled);
    void setVisible(bool visible);
};

class ScoreView {
public:
    ScoreView();
    ~ScoreView();

    bool isMultiRest() const;
    void setMultiRest(bool enabled);
    void clearMasterBars();
    // ?systems@ScoreView@core@gp@@QEBAAEBV?$vector@V?$shared_ptr@VSystemView@view@core@gp@@@std@@...
    const std::vector<std::shared_ptr<view::SystemView>>& systems() const;
    std::vector<std::shared_ptr<view::SystemView>>& mutableSystems();
    void clearSystems();
    const std::vector<std::shared_ptr<view::MasterBarView>>& masterBars() const;
    // ?applyModel@ScoreView@core@gp@@QEAAXAEBV123@@Z
    void applyModel(const ScoreView& other);
    // ?trackViewGroupCount@ScoreView@core@gp@@QEBAIXZ
    unsigned int trackViewGroupCount() const;
    TrackViewGroup& trackViewGroup(unsigned int trackIndex);
    void insertTrackViewGroup(
        unsigned int trackIndex,
        const Track* track,
        TrackView::Type type);
    void removeTrackViewGroupAtTrackIndex(unsigned int trackIndex);

private:
    alignas(16) unsigned char _opaque[65536];
};

}} // gp::core
// ──────────────────────────────────────────────────────────────────
// gp::core::Score（GPCore.dll）
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core {

enum class AntiAccent : int;
enum class Bend : int;
enum class Direction : int;
enum class DirectionMark : int;
enum class Fadding : int;
enum class Fingering : int;
enum class Fermata : int;
enum class Golpe : int;
enum class GraceType : int;
enum class Hairpin : int;
enum class Ornament : int;
enum class Rasgueado : int;
enum class ScoreProperty : int;
enum class SimileMark : int;
enum class SustainPedalState : int;
enum class Vibrato : int;
enum class WahWah : int;
enum class WhammyBar : int;
enum class Ottavia : int;
enum class TempoUnit : int;

class Automation {
public:
    enum class Type : int;

    virtual unsigned int barIndex() const;
    Type type() const;
    float position() const;
    float value() const;
    bool isVisible() const;
    bool isLinear() const;
    const std::string& text() const;
    static const std::string typeToString(Type type);
};

class ScoreViewCollection {
public:
    void clear();
};

class TempoAutomation : public Automation {
public:
    TempoUnit unit() const;
};

class AutomationContainerProxy {
public:
    virtual void getAutomations(
        std::vector<std::shared_ptr<Automation>>& output) const;
};

class NoteDynamic {
public:
    am::music::Dynamic value() const;
    std::string toString() const;
};

class alignas(8) KeySignature {
public:
    int accidentalCount() const;
private:
    unsigned char _abiStorage[16];
};

static_assert(sizeof(KeySignature) == 16, "KeySignature ABI size mismatch");
static_assert(alignof(KeySignature) == 8, "KeySignature ABI alignment mismatch");

class Harmonic {
public:
    enum class Type : int;
    enum class Fret : int;
    static float fretToFloat(Fret fret);
    static std::string typeToString(Type type);
};

class InstrumentSet {
public:
    enum class Type : int;

    static bool isBass(Type type);
    static bool isGuitar(Type type);
};

class RhythmValue {
public:
    enum class Value : int;

    Value getNoteValue() const;
    unsigned int getAugmentationDot() const;
    bool hasPrimaryTuplet() const;
    bool hasSecondaryTuplet() const;
    const std::pair<unsigned char, unsigned char>& getPrimaryTupletRatio() const;
    const std::pair<unsigned char, unsigned char>& getSecondaryTupletRatio() const;
};

class GuitarTuning {
public:
    enum class Role : int;

    unsigned int stringCount() const;
    int tuning(unsigned int stringIndex) const;
    const std::string& displayedLabel() const;
    bool labelVisible() const;
    Role role() const;
    bool isFlat() const;
    bool hasShortDroneString() const;
};

class TimeSignature {
public:
    unsigned int getNumerator() const;
    unsigned int getDenominator() const;
};

struct FermataInfo {
    Fermata type;
    unsigned char _padding0[4];
    am::utils::rational offset;
    float length;
    unsigned char _padding1[4];
};

static_assert(sizeof(FermataInfo) == 48, "FermataInfo ABI size mismatch");

class MasterBar {
public:
    struct Section {
        std::string letter;
        std::string text;
    };

    enum class TripletFeel : int;

    unsigned int index() const;
    const TimeSignature& timeSignature() const;
    bool hasRepeatStart() const;
    bool hasRepeatEnd() const;
    unsigned int repeatCount() const;
    bool hasAlternateEndings() const;
    int alternateEndingMask() const;
    bool hasDoubleBar() const;
    bool hasFreeTime() const;
    bool hasAnacrusis() const;
    bool hasFermata() const;
    const std::set<FermataInfo>& fermatas() const;
    bool hasTripletFeel() const;
    const KeySignature& concertKeySignature() const;
    const KeySignature drawingKeySignatureOfTrackAndStaff(
        unsigned int trackIndex,
        unsigned int staffIndex) const;
    bool hasExtendedAlternateEndings() const;
    bool isExtendedAlternateEndingSet(unsigned int index) const;
    bool hasSection() const;
    const Section& section() const;
    TripletFeel tripletFeel() const;
    static QString tripletFeelToQString(TripletFeel value);
};

std::string antiAccentToString(AntiAccent value);
std::string bendToString(Bend value);
std::string directionToString(Direction value);
std::string faddingToString(Fadding value);
std::string fingeringToString(Fingering value);
std::string golpeToString(Golpe value);
std::string graceTypeToString(GraceType value);
std::string hairpinToString(Hairpin value);
std::string ornamentToString(Ornament value);
std::string rasgueadoToString(Rasgueado value);
ScoreProperty stringToScoreProperty(const std::string& value);
const std::string simileMarkToString(SimileMark value);
std::string vibratoToString(Vibrato value);
std::string wahWahToString(WahWah value);
std::string whammyBarToString(WhammyBar value);
std::string ottaviaToString(Ottavia value);
std::string tempoUnitToString(TempoUnit value);

class MasterTrack : public AutomationContainerProxy {
public:
    unsigned int masterBarCount() const;
    std::shared_ptr<MasterBar> masterBar(unsigned int index) const;
    float tempoValue() const;
    bool tempoVisible() const;
    const std::string& tempoLabel() const;
    TempoUnit tempoUnit() const;
    std::set<DirectionMark> directionsAtBarIndex(int barIndex) const;
    static QString directionToQString(DirectionMark direction);
};

class Track;
class Staff;
class Bar;
class Voice;
class Beat;
class Note;

class LyricsElement {
public:
    enum class Continuation : int;
    enum class HAlignment : int;

    virtual ~LyricsElement();
    const std::string& text() const;
    std::string displayedText() const;
    Continuation extend() const;
    Continuation syllabic() const;
    HAlignment halignment() const;
    double xOffset() const;

private:
    alignas(8) unsigned char _opaque[56];
};

static_assert(sizeof(LyricsElement) == 64, "LyricsElement ABI size mismatch");

namespace diagram {

class Barre {
public:
    virtual ~Barre();
    unsigned int fret() const;
    unsigned int startString() const;
    unsigned int endString() const;

private:
    void* _data;
};

static_assert(sizeof(Barre) == 16, "Barre ABI size mismatch");

class FreeDiagram {
public:
    virtual ~FreeDiagram();
    unsigned int baseFret() const;
    unsigned int fretCount() const;
    const std::vector<Barre>& barres() const;
    virtual unsigned int stringCount() const;
};

} // namespace diagram

namespace chord {

class Fingering {
public:
    enum class Finger : int;

    Finger finger(unsigned int stringIndex, unsigned int fret) const;
    QString toString() const;
};

class Diagram : public diagram::FreeDiagram {
public:
    enum class FretValueType : int {
        Absolute = 0,
        Relative = 1,
    };

    unsigned int spanLimit() const;
    unsigned int stringCount() const override;
    bool showDiagram() const;
    bool showName() const;
    bool showFingering() const;
    unsigned int fret(
        unsigned int stringIndex,
        FretValueType valueType) const;
    std::vector<std::tuple<unsigned int, unsigned int>> frets(
        FretValueType valueType) const;
    const Fingering* fingering() const;
};

class ChordEntry {
public:
    QString name() const;
};

class DiagramEntry {
public:
    QString name() const;
    const Diagram& diagram() const;
};

class ChordCollectionItem {
public:
    const ChordEntry& entry() const;
};

class DiagramCollectionItem {
public:
    const DiagramEntry& entry() const;
};

class ChordCollection {
public:
    const ChordCollectionItem* find(const QString& id) const;
};

class DiagramCollection {
public:
    const DiagramCollectionItem* find(const QString& id) const;
    std::vector<const DiagramCollectionItem*> items() const;
};

} // namespace chord

class Note {
public:
    unsigned int index() const;
    unsigned int string() const;
    int fret() const;
    bool isPrintableInTab() const;
    bool isDead() const;
    bool isPalmMuted() const;
    bool hasLetRing() const;
    unsigned int accentFlags() const;
    bool hasAntiAccent() const;
    AntiAccent antiAccent() const;
    bool isTieOrigin() const;
    bool isTieDestination() const;
    std::shared_ptr<Note> tieOrigin() const;
    std::shared_ptr<Note> tieDestination() const;
    bool isHopoOrigin() const;
    bool isHopoDestination() const;
    std::shared_ptr<Note> hopoOrigin() const;
    std::shared_ptr<Note> hopoDestination() const;
    unsigned int slideFlags() const;
    bool hasInFromAboveSlide() const;
    bool hasInFromBelowSlide() const;
    bool hasLegatoSlide() const;
    bool hasShiftSlide() const;
    bool hasOutDownwardsSlide() const;
    bool hasOutUpwardsSlide() const;
    bool hasOutDownwardsPickScrape() const;
    bool hasOutUpwardsPickScrape() const;
    std::shared_ptr<Note> slideBegin() const;
    std::shared_ptr<Note> slideEnd() const;
    bool isBended() const;
    Bend bendType() const;
    float bendOriginOffset() const;
    float bendOriginValue() const;
    float bendMiddleOffset1() const;
    float bendMiddleOffset2() const;
    float bendMiddleValue() const;
    float bendDestinationOffset() const;
    float bendDestinationValue() const;
    bool isHarmonic() const;
    Harmonic::Type harmonicType() const;
    Harmonic::Fret harmonicFret() const;
    bool isTrilled() const;
    int trillFret() const;
    bool isTrillValid() const;
    unsigned int trillMidi() const;
    bool hasVibrato() const;
    Vibrato vibrato() const;
    bool isTapped() const;
    bool isLeftHandTapped() const;
    bool hasPizzicato() const;
    bool showStringNumber() const;
    bool hasSustainPedal() const;
    bool hasOrnament() const;
    Ornament ornament() const;
    Fingering leftHandFingering() const;
    Fingering rightHandFingering() const;
};

class Beat {
public:
    unsigned int index() const;
    bool isRest() const;
    bool isPlaceholder() const;
    bool hasSustainPedal() const;
    bool isDeadSlapped() const;
    bool isGraced() const;
    GraceType graceType() const;
    unsigned int graceBeatCount() const;
    const RhythmValue& rhythm() const;
    const am::utils::rational& offsetIgnoringGraceBeats() const;
    unsigned int noteCount() const;
    std::shared_ptr<Note> note(unsigned int index) const;
    bool hasBarre() const;
    unsigned int barreFret() const;
    unsigned int barreString() const;
    bool hasFadding() const;
    Fadding fadding() const;
    bool hasGolpe() const;
    Golpe golpe() const;
    bool hasHairpin() const;
    Hairpin hairpin() const;
    const NoteDynamic& dynamic() const;
    bool hasWahWah() const;
    WahWah wahWah() const;
    bool hasTimer() const;
    unsigned int timer() const;
    bool isSlashed() const;
    bool hasFreeText() const;
    const std::string& freeText() const;
    bool hasChord() const;
    const QString& chord() const;
    bool isBrushed() const;
    Direction brush() const;
    bool hasArpeggio() const;
    Direction arpeggio() const;
    bool isPickStroked() const;
    Direction pickStroke() const;
    bool isSlapped() const;
    bool isPopped() const;
    bool hasRasgueado() const;
    Rasgueado rasgueado() const;
    bool hasTremolo() const;
    const am::utils::rational& tremolo() const;
    bool hasWhammyBar() const;
    WhammyBar whammyBarType() const;
    bool whammyBarExtend() const;
    unsigned int whammyBarBeatCount() const;
    std::shared_ptr<Beat> whammyBarBegin() const;
    std::shared_ptr<Beat> whammyBarEnd() const;
    float whammyBarOriginOffset() const;
    float whammyBarOriginValue() const;
    float whammyBarMiddleOffset1() const;
    float whammyBarMiddleOffset2() const;
    float whammyBarMiddleValue() const;
    float whammyBarDestinationOffset() const;
    float whammyBarDestinationValue() const;
    bool hasVibratoWTremBar() const;
    Vibrato vibratoWTremBar() const;
    Ottavia ottavia() const;
    const std::array<LyricsElement, 5>& lyrics() const;
    bool isLegatoOrigin() const;
    bool isLegatoDestination() const;
    std::shared_ptr<Beat> legatoOrigin() const;
    std::shared_ptr<Beat> legatoDestination() const;
};

class Voice {
public:
    unsigned int index() const;
    unsigned int beatCount() const;
    std::shared_ptr<Beat> beat(unsigned int index) const;
};

class Bar {
public:
    unsigned int index() const;
    unsigned int staffIndex() const;
    unsigned int voiceCount() const;
    std::shared_ptr<Voice> voice(unsigned int index) const;
    std::shared_ptr<MasterBar> masterBar() const;
    Ottavia ottavia() const;
    SimileMark simileMark() const;
};

class Staff {
public:
    unsigned int index() const;
    unsigned int barCount() const;
    std::shared_ptr<Bar> bar(unsigned int index) const;
    unsigned char capoFret() const;
    bool hasPartialCapoOnString(unsigned int stringIndex) const;
    unsigned char partialCapoFretOnString(unsigned int stringIndex) const;
    unsigned char totalCapoFretOnString(unsigned int stringIndex) const;
    unsigned int openStringFret(unsigned int stringIndex) const;
    GuitarTuning& tuning() const;
    chord::ChordCollection& chordCollection() const;
    chord::DiagramCollection& diagramCollection() const;
};

class TrackBase {
public:
    const std::string& name() const;
    const std::string& shortName() const;
};

class Track : public TrackBase {
public:
    unsigned int staffCount() const;
    std::shared_ptr<Staff> staff(unsigned int index) const;
    unsigned int barCount() const;
    InstrumentSet::Type type() const;
    bool hasLetRingThroughout() const;
};

class Score {
public:
    Score();   // ??0Score@core@gp@@QEAA@XZ
    ~Score();  // ??1Score@core@gp@@QEAA@XZ  (public non-virtual)

    // ?activeView@Score@core@gp@@QEAAAEAVScoreView@23@XZ
    ScoreView& activeView();
    const ScoreView& activeView() const;
    ScoreViewCollection& views();
    // ?newStylesheet@Score@core@gp@@QEBAAEBV?$shared_ptr@VStylesheet...
    const std::shared_ptr<style::Stylesheet>& newStylesheet() const;
    unsigned int trackCount() const;
    std::shared_ptr<Track> track(unsigned int index) const;
    std::shared_ptr<MasterTrack> masterTrack() const;
    std::string property(ScoreProperty property) const;

    // ?init@Score@core@gp@@QEAAXXZ：用默认空结构初始化 score。
    void init();
    // ?load@Score@core@gp@@QEAAXAEAVFileHandle@filesystem@am@@PEAVImporter@io@23@@Z
    void load(am::filesystem::FileHandle& file, io::Importer* importer);
    // ?load@Score@core@gp@@QEAAXAEBVQString@@@Z
    void load(const QString& path);
private:
    Score(const Score&) = delete;
    Score& operator=(const Score&) = delete;
    alignas(16) unsigned char _opaque[262144];
};

}} // gp::core


// ──────────────────────────────────────────────────────────────────
// gp::core::Core：应用单例（GPCore.dll）
// 加载任何 score 前必须先调用 Core::initialize(fileSystem)。
// 如果 Core 尚未初始化，load() 内部调用 Core::instance() 时会导致
// Track::cloneTrack 读取 NULL+0x48 这类崩溃。
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core {

class Track;

class Core {
public:
    // ?initialize@Core@core@gp@@SAXPEAVFileSystem@filesystem@am@@@Z
    static void initialize(am::filesystem::FileSystem* fs);
    // ?initialize@Core@core@gp@@SAXPEAVFileSystem@filesystem@am@@AEBVQString@@@Z
    static void initialize(am::filesystem::FileSystem* fs, const QString& appDataPath);
    // ?instance@Core@core@gp@@SAAEAV123@XZ
    static Core& instance();
    // ?loadNotationPatches@Core@core@gp@@QEAAXXZ
    // 从 GPCore.dll 内嵌 Qt 资源加载 GM 乐器模板。
    // 必须在 initialize() 后调用，findGMInstrument() 才会返回非空轨道。
    void loadNotationPatches();

    // ?loadSoundBankDirectory@Core@core@gp@@QEAAXAEBVQString@@@Z
    void loadSoundBankDirectory(const QString& path);

    // ?loadSoundBankSetDirectory@Core@core@gp@@QEAAXAEBVQString@@@Z
    void loadSoundBankSetDirectory(const QString& path);

    // ?loadAdditionalSoundBanksDirectory@Core@core@gp@@QEAAXAEBVQString@@@Z
    void loadAdditionalSoundBanksDirectory(const QString& path);

    // ?findGMInstrumentPath@Core@core@gp@@QEBA?BVQString@@I@Z
    // 返回第 n 个 GM 乐器的 Qt 资源路径；未加载时返回空字符串。
    const QString findGMInstrumentPath(unsigned int midiProgram) const;
    // ?findGMInstrument@Core@core@gp@@QEBA?BV?$shared_ptr@VTrack@core@gp@@@std@@I@Z
    const std::shared_ptr<Track> findGMInstrument(unsigned int midiProgram) const;

    // ?waitLazyConf@Core@core@gp@@QEBAXXZ
    // 等待所有延迟/异步初始化完成，包括 GM 乐器表。
    void waitLazyConf() const;

    // ?importerByHandleAndExtension@Core@core@gp@@QEBAPEAVImporter@io@23@
    //   AEAVFileHandle@filesystem@am@@AEAVQString@@@Z
    io::Importer* importerByHandleAndExtension(
        am::filesystem::FileHandle& file,
        QString& extension) const;

private:
    Core();  // ??0Core@core@gp@@AEAA@XZ：私有构造函数。
};

}} // gp::core


// ──────────────────────────────────────────────────────────────────
// gp::core::LayoutHandler（GPCore.dll）
// ──────────────────────────────────────────────────────────────────
namespace gp { namespace core {

class LayoutHandler {
public:
    // ?layoutsInfo@LayoutHandler@core@gp@@SA?AU?$pair@V?$vector@HV?$allocator@H@std@@@std@@H@std@@AEBVScore@23@@Z
    static std::pair<std::vector<int>, int> layoutsInfo(const Score& score);
    // ?setLayouts@LayoutHandler@core@gp@@SAXAEAVScore@23@AEBV?$vector@HV?$allocator@H@std@@@std@@@Z
    static void setLayouts(Score& score, const std::vector<int>& layouts);
    // ?buildLayoutsOnActiveView@LayoutHandler@core@gp@@SAXAEAVScore@23@@Z
    static void buildLayoutsOnActiveView(Score& score);
    // ?buildLayouts@LayoutHandler@core@gp@@SAXAEAVScore@23@@Z
    static void buildLayouts(Score& score);
    // ?updateLayoutsOnActiveView@LayoutHandler@core@gp@@SAXAEAVScore@23@@Z
    static void updateLayoutsOnActiveView(Score& score);
};

}} // gp::core
