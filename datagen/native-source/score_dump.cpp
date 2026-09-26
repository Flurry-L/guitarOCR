#include "score_dump.h"

#include "gp_stubs.h"

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <fstream>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpomr {
namespace {

std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (unsigned char character : value) {
        switch (character) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (character < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(character)
                        << std::dec << std::setfill(' ');
                } else {
                    out << character;
                }
        }
    }
    return out.str();
}

const char* jsonBool(bool value) {
    return value ? "true" : "false";
}

template <typename Enum>
int enumCode(Enum value) {
    return static_cast<int>(value);
}

void writeRational(std::ostream& out, const am::utils::rational& value) {
    out << "[" << value.numerator() << "," << value.denominator() << "]";
}

void writeTuplet(std::ostream& out, const gp::core::RhythmValue& rhythm) {
    out << "{\"primary\":";
    if (rhythm.hasPrimaryTuplet()) {
        const auto& ratio = rhythm.getPrimaryTupletRatio();
        out << "[" << static_cast<unsigned int>(ratio.first)
            << "," << static_cast<unsigned int>(ratio.second) << "]";
    } else {
        out << "null";
    }
    out << ",\"secondary\":";
    if (rhythm.hasSecondaryTuplet()) {
        const auto& ratio = rhythm.getSecondaryTupletRatio();
        out << "[" << static_cast<unsigned int>(ratio.first)
            << "," << static_cast<unsigned int>(ratio.second) << "]";
    } else {
        out << "null";
    }
    out << "}";
}

std::string trackSemanticId(unsigned int trackIndex) {
    return "track:" + std::to_string(trackIndex);
}

std::string staffSemanticId(unsigned int trackIndex, unsigned int staffIndex) {
    return trackSemanticId(trackIndex) + "/staff:" + std::to_string(staffIndex);
}

std::string measureSemanticId(
    unsigned int trackIndex,
    unsigned int staffIndex,
    unsigned int measureIndex) {
    return staffSemanticId(trackIndex, staffIndex)
        + "/measure:" + std::to_string(measureIndex);
}

std::string voiceSemanticId(
    unsigned int trackIndex,
    unsigned int staffIndex,
    unsigned int measureIndex,
    unsigned int voiceIndex) {
    return measureSemanticId(trackIndex, staffIndex, measureIndex)
        + "/voice:" + std::to_string(voiceIndex);
}

std::string eventSemanticId(
    unsigned int trackIndex,
    unsigned int staffIndex,
    unsigned int measureIndex,
    unsigned int voiceIndex,
    unsigned int eventIndex) {
    return voiceSemanticId(trackIndex, staffIndex, measureIndex, voiceIndex)
        + "/event:" + std::to_string(eventIndex);
}

std::string noteSemanticId(
    const std::string& eventId,
    unsigned int noteIndex) {
    return eventId + "/note:" + std::to_string(noteIndex);
}

using BeatSemanticRefs = std::map<const gp::core::Beat*, std::string>;
using NoteSemanticRefs = std::map<const gp::core::Note*, std::string>;

struct SemanticRefs {
    BeatSemanticRefs beats;
    NoteSemanticRefs notes;
};

SemanticRefs collectSemanticRefs(
    const gp::core::Track& track,
    unsigned int trackIndex) {
    SemanticRefs refs;
    for (unsigned int staffPosition = 0;
         staffPosition < track.staffCount();
         ++staffPosition) {
        const auto staff = track.staff(staffPosition);
        if (!staff) continue;
        for (unsigned int barPosition = 0;
             barPosition < staff->barCount();
             ++barPosition) {
            const auto bar = staff->bar(barPosition);
            if (!bar) continue;
            for (unsigned int voicePosition = 0;
                 voicePosition < bar->voiceCount();
                 ++voicePosition) {
                const auto voice = bar->voice(voicePosition);
                if (!voice) continue;
                for (unsigned int beatPosition = 0;
                     beatPosition < voice->beatCount();
                     ++beatPosition) {
                    const auto beat = voice->beat(beatPosition);
                    if (!beat) continue;
                    const std::string semanticId = eventSemanticId(
                        trackIndex,
                        staff->index(),
                        bar->index(),
                        voice->index(),
                        beat->index());
                    const auto inserted = refs.beats.emplace(beat.get(), semanticId);
                    if (!inserted.second && inserted.first->second != semanticId) {
                        throw std::runtime_error(
                            "one beat belongs to multiple official score locations");
                    }
                    for (unsigned int notePosition = 0;
                         notePosition < beat->noteCount();
                         ++notePosition) {
                        const auto note = beat->note(notePosition);
                        if (!note) continue;
                        const std::string noteId = noteSemanticId(
                            semanticId,
                            note->index());
                        const auto noteInserted = refs.notes.emplace(
                            note.get(),
                            noteId);
                        if (!noteInserted.second
                                && noteInserted.first->second != noteId) {
                            throw std::runtime_error(
                                "one note belongs to multiple official score locations");
                        }
                    }
                }
            }
        }
    }
    return refs;
}

std::string officialNameUtf8(const std::string& value) {
    return value;
}

std::string utf8(const QString& value) {
    const QByteArray bytes = value.toUtf8();
    return std::string(bytes.constData(), static_cast<size_t>(bytes.size()));
}

void writeOptionalString(std::ostream& out, const std::string& value);

template <typename NameFunction>
std::string optionalOfficialName(NameFunction&& function) noexcept {
    try {
        return officialNameUtf8(function());
    } catch (...) {
        return {};
    }
}

template <typename NameFunction>
void writeOptionalOfficialName(
    std::ostream& out,
    NameFunction&& function) {
    writeOptionalString(out, optionalOfficialName(function));
}

template <typename ValueFunction, typename NameFunction>
void writeOptionalNamedEnumFields(
    std::ostream& out,
    const char* prefix,
    bool present,
    ValueFunction&& valueFunction,
    NameFunction&& nameFunction) {
    out << ",\"" << prefix << "_code\":";
    if (!present) {
        out << "0,\"" << prefix << "_name\":null";
        return;
    }
    const auto value = valueFunction();
    out << enumCode(value) << ",\"" << prefix << "_name\":";
    writeOptionalOfficialName(out, [&]() { return nameFunction(value); });
}

using VisibilityGetter = gp::core::view::Visibility (*)(
    const gp::core::style::Stylesheet&);
using FormattedTextGetter = am::painting::FormattedText (*)(
    const gp::core::style::Stylesheet&);

struct StyledTextField {
    const char* name;
    VisibilityGetter visibility;
    FormattedTextGetter formattedText;
};

bool isVisible(gp::core::view::Visibility value) {
    return value == gp::core::view::Visibility::Visible;
}

template <size_t FieldCount>
void writeStyledTextFields(
    std::ostream& out,
    const gp::core::style::Stylesheet& stylesheet,
    const StyledTextField (&fields)[FieldCount]) {
    out << "{";
    for (size_t index = 0; index < FieldCount; ++index) {
        if (index) out << ",";
        const auto formattedText = fields[index].formattedText(stylesheet);
        out << "\"" << fields[index].name << "\":{\"visible\":"
            << jsonBool(isVisible(fields[index].visibility(stylesheet)))
            << ",\"formatted_text\":\""
            << escapeJson(formattedText.text()) << "\"}";
    }
    out << "}";
}

void writePageHeader(
    std::ostream& out,
    const gp::core::style::Stylesheet& stylesheet,
    VisibilityGetter visibility,
    const StyledTextField& field) {
    const auto formattedText = field.formattedText(stylesheet);
    out << "{\"visible\":" << jsonBool(isVisible(visibility(stylesheet)))
        << ",\"field\":{\"visible\":"
        << jsonBool(isVisible(field.visibility(stylesheet)))
        << ",\"formatted_text\":\""
        << escapeJson(formattedText.text()) << "\"}}";
}

template <size_t FieldCount>
void writePageFooter(
    std::ostream& out,
    const gp::core::style::Stylesheet& stylesheet,
    VisibilityGetter visibility,
    const StyledTextField (&fields)[FieldCount]) {
    out << "{\"visible\":" << jsonBool(isVisible(visibility(stylesheet)))
        << ",\"fields\":";
    writeStyledTextFields(out, stylesheet, fields);
    out << "}";
}

void writeScoreMetadata(
    std::ostream& out,
    const gp::core::Score& score,
    const gp::core::style::Stylesheet& stylesheet) {
    struct Field {
        const char* outputName;
        const char* propertyName;
    };
    static const Field kFields[] = {
        {"title", "TITLE"},
        {"subtitle", "SUBTITLE"},
        {"artist", "ARTIST"},
        {"album", "ALBUM"},
        {"words", "WORDS"},
        {"music", "MUSIC"},
        {"words_and_music", "WORDSANDMUSIC"},
        {"copyright", "COPYRIGHT"},
        {"tabber", "TABBER"},
        {"instructions", "INSTRUCTIONS"},
        {"notice", "NOTICE"},
    };

    std::map<std::string, std::string> values;
    for (size_t index = 0; index < sizeof(kFields) / sizeof(kFields[0]); ++index) {
        const auto property = gp::core::stringToScoreProperty(
            kFields[index].propertyName);
        values.emplace(kFields[index].outputName, score.property(property));
    }

    out << "{\"properties\":{";
    for (size_t index = 0; index < sizeof(kFields) / sizeof(kFields[0]); ++index) {
        if (index) out << ",";
        out << "\"" << kFields[index].outputName << "\":\""
            << escapeJson(values.at(kFields[index].outputName)) << "\"";
    }

    using namespace gp::core::style;
    const StyledTextField headerFields[] = {
        {"title",
         scoreFirstPageHeaderTitleVisibilityValue,
         scoreFirstPageHeaderTitleFormattedTextValue},
        {"subtitle",
         scoreFirstPageHeaderSubtitleVisibilityValue,
         scoreFirstPageHeaderSubtitleFormattedTextValue},
        {"artist",
         scoreFirstPageHeaderArtistVisibilityValue,
         scoreFirstPageHeaderArtistFormattedTextValue},
        {"album",
         scoreFirstPageHeaderAlbumVisibilityValue,
         scoreFirstPageHeaderAlbumFormattedTextValue},
        {"words",
         scoreFirstPageHeaderLyricsVisibilityValue,
         scoreFirstPageHeaderLyricsFormattedTextValue},
        {"music",
         scoreFirstPageHeaderMusicVisibilityValue,
         scoreFirstPageHeaderMusicFormattedTextValue},
        {"words_and_music",
         scoreFirstPageHeaderWordsAndMusicVisibilityValue,
         scoreFirstPageHeaderWordsAndMusicFormattedTextValue},
        {"tabber",
         scoreFirstPageHeaderTabberVisibilityValue,
         scoreFirstPageHeaderTabberFormattedTextValue},
    };
    const StyledTextField evenPageHeaderField = {
        "field",
        scoreEvenPageHeaderFieldVisibilityValue,
        scoreEvenPageHeaderFieldFormattedTextValue,
    };
    const StyledTextField oddPageHeaderField = {
        "field",
        scoreOddPageHeaderFieldVisibilityValue,
        scoreOddPageHeaderFieldFormattedTextValue,
    };
    const StyledTextField firstPageFooterFields[] = {
        {"copyright",
         scoreFirstPageFooterCopyrightVisibilityValue,
         scoreFirstPageFooterCopyrightFormattedTextValue},
        {"copyright2",
         scoreFirstPageFooterCopyright2VisibilityValue,
         scoreFirstPageFooterCopyright2FormattedTextValue},
        {"page_number",
         scoreFirstPageFooterPageNumberVisibilityValue,
         scoreFirstPageFooterPageNumberFormattedTextValue},
    };
    const StyledTextField evenPageFooterFields[] = {
        {"copyright",
         scoreEvenPageFooterCopyrightVisibilityValue,
         scoreEvenPageFooterCopyrightFormattedTextValue},
        {"copyright2",
         scoreEvenPageFooterCopyright2VisibilityValue,
         scoreEvenPageFooterCopyright2FormattedTextValue},
        {"page_number",
         scoreEvenPageFooterPageNumberVisibilityValue,
         scoreEvenPageFooterPageNumberFormattedTextValue},
    };
    const StyledTextField oddPageFooterFields[] = {
        {"copyright",
         scoreOddPageFooterCopyrightVisibilityValue,
         scoreOddPageFooterCopyrightFormattedTextValue},
        {"copyright2",
         scoreOddPageFooterCopyright2VisibilityValue,
         scoreOddPageFooterCopyright2FormattedTextValue},
        {"page_number",
         scoreOddPageFooterPageNumberVisibilityValue,
         scoreOddPageFooterPageNumberFormattedTextValue},
    };

    out << "},\"first_page_header\":";
    writeStyledTextFields(out, stylesheet, headerFields);
    out << ",\"even_page_header\":";
    writePageHeader(
        out,
        stylesheet,
        scoreEvenPageHeaderVisibilityValue,
        evenPageHeaderField);
    out << ",\"odd_page_header\":";
    writePageHeader(
        out,
        stylesheet,
        scoreOddPageHeaderVisibilityValue,
        oddPageHeaderField);
    out << ",\"first_page_footer\":";
    writePageFooter(
        out,
        stylesheet,
        scoreFirstPageFooterVisibilityValue,
        firstPageFooterFields);
    out << ",\"even_page_footer\":";
    writePageFooter(
        out,
        stylesheet,
        scoreEvenPageFooterVisibilityValue,
        evenPageFooterFields);
    out << ",\"odd_page_footer\":";
    writePageFooter(
        out,
        stylesheet,
        scoreOddPageFooterVisibilityValue,
        oddPageFooterFields);
    out << "}";
}

void writeDiagramStyle(
    std::ostream& out,
    const gp::core::style::Stylesheet& stylesheet) {
    using namespace gp::core::style;
    const auto headerVisibility = scoreHeaderDiagramVisibilityValue(stylesheet);
    const auto scoreVisibility = scoreScoreDiagramVisibilityValue(stylesheet);
    out << "{\"header_visibility_code\":" << enumCode(headerVisibility)
        << ",\"header_visible\":" << jsonBool(isVisible(headerVisibility))
        << ",\"header_chord_name_display_code\":"
        << enumCode(scoreHeaderDiagramChordNameDisplayValue(stylesheet))
        << ",\"header_fingering_display_code\":"
        << enumCode(scoreHeaderDiagramFingeringDisplayValue(stylesheet))
        << ",\"score_visibility_code\":" << enumCode(scoreVisibility)
        << ",\"score_visible\":" << jsonBool(isVisible(scoreVisibility))
        << ",\"score_chord_name_display_code\":"
        << enumCode(scoreScoreDiagramChordNameDisplayValue(stylesheet))
        << ",\"score_fingering_display_code\":"
        << enumCode(scoreScoreDiagramFingeringDisplayValue(stylesheet))
        << "}";
}

void writeOptionalString(std::ostream& out, const std::string& value) {
    if (value.empty()) {
        out << "null";
    } else {
        out << "\"" << escapeJson(value) << "\"";
    }
}

void writeTempoAutomations(
    std::ostream& out,
    const gp::core::MasterTrack& masterTrack) {
    std::vector<std::shared_ptr<gp::core::Automation>> automations;
    masterTrack.gp::core::AutomationContainerProxy::getAutomations(automations);

    out << "[";
    bool first = true;
    for (const auto& automation : automations) {
        if (!automation) continue;
        const auto type = automation->type();
        const std::string typeName = gp::core::Automation::typeToString(type);
        if (typeName != "Tempo") continue;

        const auto* tempo = static_cast<const gp::core::TempoAutomation*>(
            automation.get());
        if (!first) out << ",";
        first = false;
        out << "{\"bar_index\":"
            << automation->gp::core::Automation::barIndex()
            << ",\"position\":" << automation->position()
            << ",\"value\":" << automation->value()
            << ",\"unit_code\":" << enumCode(tempo->unit())
            << ",\"unit_name\":";
        writeOptionalOfficialName(
            out, [&]() { return gp::core::tempoUnitToString(tempo->unit()); });
        out << ",\"visible\":" << jsonBool(automation->isVisible())
            << ",\"linear\":" << jsonBool(automation->isLinear())
            << ",\"text\":\"" << escapeJson(automation->text()) << "\"}";
    }
    out << "]";
}

void writeBend(std::ostream& out, const gp::core::Note& note) {
    const auto type = note.bendType();
    const std::string typeName = optionalOfficialName(
        [&]() { return gp::core::bendToString(type); });
    out << "{\"type_code\":" << enumCode(type)
        << ",\"type\":";
    writeOptionalString(out, typeName);
    out << ",\"origin\":[" << note.bendOriginOffset() << ","
        << note.bendOriginValue() << "]"
        << ",\"middle\":[" << note.bendMiddleOffset1() << ","
        << note.bendMiddleOffset2() << "," << note.bendMiddleValue() << "]"
        << ",\"destination\":[" << note.bendDestinationOffset() << ","
        << note.bendDestinationValue() << "]}";
}

void writeBeatRef(
    std::ostream& out,
    const std::shared_ptr<gp::core::Beat>& beat,
    const BeatSemanticRefs& refs) {
    if (!beat) {
        out << "null";
        return;
    }
    const auto found = refs.find(beat.get());
    if (found == refs.end()) {
        out << "null";
        return;
    }
    out << "\"" << escapeJson(found->second) << "\"";
}

void writeNoteRef(
    std::ostream& out,
    const std::shared_ptr<gp::core::Note>& note,
    const NoteSemanticRefs& refs) {
    if (!note) {
        out << "null";
        return;
    }
    const auto found = refs.find(note.get());
    if (found == refs.end()) {
        out << "null";
        return;
    }
    out << "\"" << escapeJson(found->second) << "\"";
}

void writeWhammy(
    std::ostream& out,
    const gp::core::Beat& beat) {
    const auto type = beat.whammyBarType();
    const std::string typeName = optionalOfficialName(
        [&]() { return gp::core::whammyBarToString(type); });
    out << "{\"type_code\":" << enumCode(type)
        << ",\"type\":";
    writeOptionalString(out, typeName);
    out << ",\"origin\":[" << beat.whammyBarOriginOffset() << ","
        << beat.whammyBarOriginValue() << "]"
        << ",\"middle\":[" << beat.whammyBarMiddleOffset1() << ","
        << beat.whammyBarMiddleOffset2() << "," << beat.whammyBarMiddleValue()
        << "]"
        << ",\"destination\":[" << beat.whammyBarDestinationOffset() << ","
        << beat.whammyBarDestinationValue() << "]}";
}

void writeNote(
    std::ostream& out,
    const gp::core::Note& note,
    const std::string& eventId,
    const NoteSemanticRefs& noteRefs) {
    const unsigned int nativeStringIndex = note.string();
    const unsigned int accentFlags = note.accentFlags();
    const unsigned int slideFlags = note.slideFlags();
    out << "{\"semantic_id\":\"" << escapeJson(
            noteSemanticId(eventId, note.index())) << "\""
        << ",\"note_index\":" << note.index()
        << ",\"native_string_index\":" << nativeStringIndex
        << ",\"fret\":" << note.fret()
        << ",\"printable_in_tab\":" << jsonBool(note.isPrintableInTab())
        << ",\"dead\":" << jsonBool(note.isDead())
        << ",\"palm_muted\":" << jsonBool(note.isPalmMuted())
        << ",\"let_ring\":" << jsonBool(note.hasLetRing())
        << ",\"accent_flags\":" << accentFlags
        << ",\"anti_accent\":" << jsonBool(note.hasAntiAccent())
        << ",\"anti_accent_code\":" << enumCode(note.antiAccent())
        << ",\"anti_accent_name\":";
    if (note.hasAntiAccent()) {
        writeOptionalOfficialName(
            out,
            [&]() { return gp::core::antiAccentToString(note.antiAccent()); });
    } else {
        out << "null";
    }
    out
        << ",\"tie_origin\":" << jsonBool(note.isTieOrigin())
        << ",\"tie_destination\":" << jsonBool(note.isTieDestination())
        << ",\"tie_origin_ref\":";
    writeNoteRef(out, note.tieOrigin(), noteRefs);
    out << ",\"tie_destination_ref\":";
    writeNoteRef(out, note.tieDestination(), noteRefs);
    out << ",\"hopo_origin\":" << jsonBool(note.isHopoOrigin())
        << ",\"hopo_destination\":" << jsonBool(note.isHopoDestination())
        << ",\"hopo_origin_ref\":";
    writeNoteRef(out, note.hopoOrigin(), noteRefs);
    out << ",\"hopo_destination_ref\":";
    writeNoteRef(out, note.hopoDestination(), noteRefs);
    out << ",\"slide_begin_ref\":";
    writeNoteRef(out, note.slideBegin(), noteRefs);
    out << ",\"slide_end_ref\":";
    writeNoteRef(out, note.slideEnd(), noteRefs);
    out << ",\"slide_flags\":" << slideFlags
        << ",\"slides\":{"
        << "\"in_above\":" << jsonBool(note.hasInFromAboveSlide())
        << ",\"in_below\":" << jsonBool(note.hasInFromBelowSlide())
        << ",\"legato\":" << jsonBool(note.hasLegatoSlide())
        << ",\"shift\":" << jsonBool(note.hasShiftSlide())
        << ",\"out_down\":" << jsonBool(note.hasOutDownwardsSlide())
        << ",\"out_up\":" << jsonBool(note.hasOutUpwardsSlide())
        << ",\"pick_scrape_down\":"
        << jsonBool(note.hasOutDownwardsPickScrape())
        << ",\"pick_scrape_up\":"
        << jsonBool(note.hasOutUpwardsPickScrape()) << "}"
        << ",\"bend\":";
    if (note.isBended()) {
        writeBend(out, note);
    } else {
        out << "null";
    }
    out << ",\"harmonic\":";
    if (note.isHarmonic()) {
        const auto harmonicType = note.harmonicType();
        const auto harmonicFret = note.harmonicFret();
        const std::string harmonicTypeName = optionalOfficialName(
            [&]() { return gp::core::Harmonic::typeToString(harmonicType); });
        out << "{\"type_code\":" << enumCode(harmonicType)
            << ",\"type\":";
        writeOptionalString(out, harmonicTypeName);
        out << ",\"fret_code\":" << enumCode(harmonicFret)
            << ",\"touch_offset\":"
            << gp::core::Harmonic::fretToFloat(harmonicFret)
            << "}";
    } else {
        out << "null";
    }
    const bool trillPresent = note.isTrilled();
    const bool trillValid = note.isTrillValid();
    const bool vibratoPresent = note.hasVibrato();
    const bool ornamentPresent = note.hasOrnament();
    out << ",\"trill_present\":" << jsonBool(trillPresent)
        << ",\"trill_fret\":";
    if (trillPresent) {
        out << note.trillFret();
    } else {
        out << "null";
    }
    out << ",\"trill_valid\":" << jsonBool(trillValid)
        << ",\"trill_midi\":";
    if (trillValid) {
        out << note.trillMidi();
    } else {
        out << "null";
    }
    out << ",\"vibrato_present\":" << jsonBool(vibratoPresent);
    writeOptionalNamedEnumFields(
        out,
        "vibrato",
        vibratoPresent,
        [&]() { return note.vibrato(); },
        [](gp::core::Vibrato value) {
            return gp::core::vibratoToString(value);
        });
    out << ",\"tapped\":" << jsonBool(note.isTapped())
        << ",\"left_hand_tapped\":" << jsonBool(note.isLeftHandTapped())
        << ",\"pizzicato\":" << jsonBool(note.hasPizzicato())
        << ",\"show_string_number\":" << jsonBool(note.showStringNumber())
        << ",\"sustain_pedal\":" << jsonBool(note.hasSustainPedal())
        << ",\"ornament_present\":" << jsonBool(ornamentPresent);
    writeOptionalNamedEnumFields(
        out,
        "ornament",
        ornamentPresent,
        [&]() { return note.ornament(); },
        [](gp::core::Ornament value) {
            return gp::core::ornamentToString(value);
        });
    const auto leftFingering = note.leftHandFingering();
    const auto rightFingering = note.rightHandFingering();
    out << ",\"left_fingering_code\":" << enumCode(leftFingering)
        << ",\"left_fingering_name\":";
    writeOptionalOfficialName(
        out,
        [&]() { return gp::core::fingeringToString(leftFingering); });
    out << ",\"right_fingering_code\":" << enumCode(rightFingering)
        << ",\"right_fingering_name\":";
    writeOptionalOfficialName(
        out,
        [&]() { return gp::core::fingeringToString(rightFingering); });
    out << "}";
}

std::string displayedChordName(
    const gp::core::Beat& beat,
    const gp::core::Staff& staff) noexcept {
    try {
        const QString& chordId = beat.chord();
        if (const auto* item = staff.chordCollection().find(chordId)) {
            return utf8(item->entry().name());
        }
        if (const auto* item = staff.diagramCollection().find(chordId)) {
            return utf8(item->entry().name());
        }
    } catch (...) {
        return {};
    }
    return {};
}

void writeLyrics(std::ostream& out, const gp::core::Beat& beat) {
    out << "[";
    bool first = true;
    const auto& lyrics = beat.lyrics();
    for (size_t lineIndex = 0; lineIndex < lyrics.size(); ++lineIndex) {
        const auto& lyric = lyrics[lineIndex];
        const std::string& text = lyric.text();
        const std::string displayedText = lyric.displayedText();
        const int extendCode = enumCode(lyric.extend());
        const int syllabicCode = enumCode(lyric.syllabic());
        if (text.empty()
            && displayedText.empty()
            && extendCode == 0
            && syllabicCode < 2) {
            continue;
        }
        if (!first) out << ",";
        first = false;
        out << "{\"line_index\":" << lineIndex
            << ",\"text\":\"" << escapeJson(text) << "\""
            << ",\"displayed_text\":\"" << escapeJson(displayedText) << "\""
            << ",\"extend_code\":" << extendCode
            << ",\"syllabic_code\":" << syllabicCode
            << ",\"h_alignment_code\":" << enumCode(lyric.halignment())
            << ",\"x_offset\":" << lyric.xOffset() << "}";
    }
    out << "]";
}

void writeDiagramEntry(
    std::ostream& out,
    const gp::core::chord::DiagramEntry& entry) {
    const auto& diagram = entry.diagram();
    const unsigned int stringCount =
        diagram.gp::core::chord::Diagram::stringCount();
    if (stringCount == 0) {
        throw std::runtime_error("official chord diagram has no strings");
    }
    const auto* fingering = diagram.fingering();
    out << "{\"name\":\"" << escapeJson(utf8(entry.name())) << "\""
        << ",\"string_count\":" << stringCount
        << ",\"base_fret\":" << diagram.baseFret()
        << ",\"fret_count\":" << diagram.fretCount()
        << ",\"span_limit\":" << diagram.spanLimit()
        << ",\"show_diagram\":" << jsonBool(diagram.showDiagram())
        << ",\"show_name\":" << jsonBool(diagram.showName())
        << ",\"show_fingering\":" << jsonBool(diagram.showFingering())
        << ",\"barres\":[";
    const auto& barres = diagram.barres();
    for (size_t barreIndex = 0; barreIndex < barres.size(); ++barreIndex) {
        if (barreIndex) out << ",";
        const auto& barre = barres[barreIndex];
        out << "{\"relative_fret\":" << barre.fret()
            << ",\"start_native_string_index\":" << barre.startString()
            << ",\"end_native_string_index\":" << barre.endString()
            << "}";
    }
    out << "],\"strings_low_to_high\":[";
    for (unsigned int stringIndex = 0;
         stringIndex < stringCount;
         ++stringIndex) {
        if (stringIndex) out << ",";
        const unsigned int fret = diagram.fret(
            stringIndex,
            gp::core::chord::Diagram::FretValueType::Relative);
        out << "{\"native_string_index\":" << stringIndex
            << ",\"relative_fret\":";
        if (fret == std::numeric_limits<unsigned int>::max()) {
            out << "null";
        } else {
            out << fret;
        }
        out << ",\"finger_code\":";
        if (fingering) {
            out << enumCode(fingering->finger(stringIndex, fret));
        } else {
            out << "null";
        }
        out << "}";
    }
    out << "]}";
}

void writeChordDiagram(
    std::ostream& out,
    const gp::core::Beat& beat,
    const gp::core::Staff& staff) {
    try {
        const auto* item = staff.diagramCollection().find(beat.chord());
        if (!item) {
            out << "null";
            return;
        }
        writeDiagramEntry(out, item->entry());
    } catch (...) {
        out << "null";
    }
}

void writeDiagramCollection(
    std::ostream& out,
    const gp::core::Staff& staff) {
    try {
        const auto items = staff.diagramCollection().items();
        std::ostringstream payload;
        payload << "[";
        for (size_t index = 0; index < items.size(); ++index) {
            if (!items[index]) {
                out << "null";
                return;
            }
            if (index) payload << ",";
            writeDiagramEntry(payload, items[index]->entry());
        }
        payload << "]";
        out << payload.str();
    } catch (...) {
        out << "null";
    }
}

void writeBeat(
    std::ostream& out,
    const gp::core::Beat& beat,
    const gp::core::Staff& staff,
    const BeatSemanticRefs& beatRefs,
    const NoteSemanticRefs& noteRefs,
    unsigned int trackIndex,
    unsigned int staffIndex,
    unsigned int measureIndex,
    unsigned int voiceIndex) {
    const auto& rhythm = beat.rhythm();
    const std::string semanticId = eventSemanticId(
        trackIndex,
        staffIndex,
        measureIndex,
        voiceIndex,
        beat.index());
    out << "{\"semantic_id\":\"" << escapeJson(semanticId) << "\""
        << ",\"event_index\":" << beat.index()
        << ",\"rest\":" << jsonBool(beat.isRest())
        << ",\"placeholder\":" << jsonBool(beat.isPlaceholder())
        << ",\"sustain_pedal\":" << jsonBool(beat.hasSustainPedal())
        << ",\"dead_slapped\":" << jsonBool(beat.isDeadSlapped())
        << ",\"grace\":" << jsonBool(beat.isGraced())
        << ",\"grace_type_code\":" << enumCode(beat.graceType())
        << ",\"grace_type_name\":";
    if (beat.isGraced()) {
        writeOptionalOfficialName(
            out,
            [&]() { return gp::core::graceTypeToString(beat.graceType()); });
    } else {
        out << "null";
    }
    const auto beatOttavia = beat.ottavia();
    const bool ottaviaPresent = enumCode(beatOttavia) != 0;
    out << ",\"grace_beat_count\":" << beat.graceBeatCount()
        << ",\"ottavia_present\":" << jsonBool(ottaviaPresent);
    writeOptionalNamedEnumFields(
        out,
        "ottavia",
        ottaviaPresent,
        [&]() { return beatOttavia; },
        [](gp::core::Ottavia value) {
            return gp::core::ottaviaToString(value);
        });
    out
        << ",\"offset\":";
    writeRational(out, beat.offsetIgnoringGraceBeats());
    out << ",\"rhythm\":{\"note_value_code\":"
        << enumCode(rhythm.getNoteValue())
        << ",\"augmentation_dots\":" << rhythm.getAugmentationDot()
        << ",\"tuplets\":";
    writeTuplet(out, rhythm);
    const bool barrePresent = beat.hasBarre();
    const bool fadePresent = beat.hasFadding();
    const bool hairpinPresent = beat.hasHairpin();
    out << "}"
        << ",\"barre_present\":" << jsonBool(barrePresent)
        << ",\"barre_string\":";
    if (barrePresent) {
        out << beat.barreString();
    } else {
        out << "null";
    }
    out << ",\"barre_fret\":";
    if (barrePresent) {
        out << beat.barreFret();
    } else {
        out << "null";
    }
    out << ",\"fade_present\":" << jsonBool(fadePresent);
    writeOptionalNamedEnumFields(
        out,
        "fade",
        fadePresent,
        [&]() { return beat.fadding(); },
        [](gp::core::Fadding value) {
            return gp::core::faddingToString(value);
        });
    out << ",\"hairpin_present\":" << jsonBool(hairpinPresent);
    writeOptionalNamedEnumFields(
        out,
        "hairpin",
        hairpinPresent,
        [&]() { return beat.hairpin(); },
        [](gp::core::Hairpin value) {
            return gp::core::hairpinToString(value);
        });
    const auto& dynamic = beat.dynamic();
    out << ",\"dynamic\":{\"code\":" << enumCode(dynamic.value())
        << ",\"name\":";
    writeOptionalOfficialName(out, [&]() { return dynamic.toString(); });
    const bool golpePresent = beat.hasGolpe();
    const bool wahPresent = beat.hasWahWah();
    out << "}"
        << ",\"golpe_present\":" << jsonBool(golpePresent);
    writeOptionalNamedEnumFields(
        out,
        "golpe",
        golpePresent,
        [&]() { return beat.golpe(); },
        [](gp::core::Golpe value) {
            return gp::core::golpeToString(value);
        });
    out << ",\"wah_present\":" << jsonBool(wahPresent);
    writeOptionalNamedEnumFields(
        out,
        "wah",
        wahPresent,
        [&]() { return beat.wahWah(); },
        [](gp::core::WahWah value) {
            return gp::core::wahWahToString(value);
        });
    const bool timerPresent = beat.hasTimer();
    const bool freeTextPresent = beat.hasFreeText();
    const bool chordPresent = beat.hasChord();
    out << ",\"slashed\":" << jsonBool(beat.isSlashed())
        << ",\"timer_present\":" << jsonBool(timerPresent)
        << ",\"timer\":";
    if (timerPresent) {
        out << beat.timer();
    } else {
        out << "null";
    }
    out << ",\"free_text_present\":" << jsonBool(freeTextPresent)
        << ",\"free_text\":";
    if (freeTextPresent) {
        out << "\"" << escapeJson(beat.freeText()) << "\"";
    } else {
        out << "null";
    }
    out << ",\"chord_present\":" << jsonBool(chordPresent)
        << ",\"chord\":";
    if (chordPresent) {
        writeOptionalString(out, displayedChordName(beat, staff));
    } else {
        out << "null";
    }
    out << ",\"chord_diagram\":";
    if (chordPresent) {
        writeChordDiagram(out, beat, staff);
    } else {
        out << "null";
    }
    out << ",\"lyrics\":";
    writeLyrics(out, beat);
    out << ",\"legato_origin\":" << jsonBool(beat.isLegatoOrigin())
        << ",\"legato_destination\":" << jsonBool(beat.isLegatoDestination())
        << ",\"legato_origin_ref\":";
    writeBeatRef(out, beat.legatoOrigin(), beatRefs);
    out << ",\"legato_destination_ref\":";
    writeBeatRef(out, beat.legatoDestination(), beatRefs);
    const bool isBrushed = beat.isBrushed();
    const bool hasArpeggio = beat.hasArpeggio();
    out
        << ",\"is_brushed\":" << jsonBool(isBrushed)
        << ",\"brush_code\":"
        << (isBrushed ? enumCode(beat.brush()) : 0)
        << ",\"brush_name\":";
    if (isBrushed) {
        writeOptionalOfficialName(
            out,
            [&]() { return gp::core::directionToString(beat.brush()); });
    } else {
        out << "null";
    }
    out
        << ",\"has_arpeggio\":" << jsonBool(hasArpeggio)
        << ",\"arpeggio_code\":"
        << (hasArpeggio ? enumCode(beat.arpeggio()) : 0)
        << ",\"arpeggio_name\":";
    if (hasArpeggio) {
        writeOptionalOfficialName(
            out,
            [&]() { return gp::core::directionToString(beat.arpeggio()); });
    } else {
        out << "null";
    }
    const bool pickStrokePresent = beat.isPickStroked();
    const bool rasgueadoPresent = beat.hasRasgueado();
    out << ",\"pick_stroke_present\":" << jsonBool(pickStrokePresent);
    writeOptionalNamedEnumFields(
        out,
        "pick_stroke",
        pickStrokePresent,
        [&]() { return beat.pickStroke(); },
        [](gp::core::Direction value) {
            return gp::core::directionToString(value);
        });
    out
        << ",\"slapped\":" << jsonBool(beat.isSlapped())
        << ",\"popped\":" << jsonBool(beat.isPopped())
        << ",\"rasgueado_present\":" << jsonBool(rasgueadoPresent);
    writeOptionalNamedEnumFields(
        out,
        "rasgueado",
        rasgueadoPresent,
        [&]() { return beat.rasgueado(); },
        [](gp::core::Rasgueado value) {
            return gp::core::rasgueadoToString(value);
        });
    out
        << ",\"tremolo\":";
    if (beat.hasTremolo()) {
        writeRational(out, beat.tremolo());
    } else {
        out << "null";
    }
    out
        << ",\"whammy\":";
    if (beat.hasWhammyBar()) {
        writeWhammy(out, beat);
    } else {
        out << "null";
    }
    out << ",\"whammy_extend\":" << jsonBool(beat.whammyBarExtend())
        << ",\"whammy_beat_count\":" << beat.whammyBarBeatCount()
        << ",\"whammy_begin_ref\":";
    writeBeatRef(out, beat.whammyBarBegin(), beatRefs);
    out << ",\"whammy_end_ref\":";
    writeBeatRef(out, beat.whammyBarEnd(), beatRefs);
    const bool tremoloBarVibratoPresent = beat.hasVibratoWTremBar();
    out << ",\"tremolo_bar_vibrato_present\":"
        << jsonBool(tremoloBarVibratoPresent);
    writeOptionalNamedEnumFields(
        out,
        "tremolo_bar_vibrato",
        tremoloBarVibratoPresent,
        [&]() { return beat.vibratoWTremBar(); },
        [](gp::core::Vibrato value) {
            return gp::core::vibratoToString(value);
        });
    out
        << ",\"notes\":[";
    for (unsigned int noteIndex = 0; noteIndex < beat.noteCount(); ++noteIndex) {
        if (noteIndex) out << ",";
        const auto note = beat.note(noteIndex);
        if (note) {
            writeNote(out, *note, semanticId, noteRefs);
        } else {
            out << "null";
        }
    }
    out << "]}";
}

void writeMasterBar(
    std::ostream& out,
    const gp::core::MasterBar& masterBar,
    const gp::core::MasterTrack& masterTrack) {
    const auto& signature = masterBar.timeSignature();
    const std::set<gp::core::DirectionMark> directions =
        masterTrack.directionsAtBarIndex(static_cast<int>(masterBar.index()));
    const bool sectionPresent = masterBar.hasSection();
    const bool tripletFeelPresent = masterBar.hasTripletFeel();
    out << "\"master_measure_index\":" << masterBar.index()
        << ",\"time_signature\":{\"numerator\":"
        << signature.getNumerator()
        << ",\"denominator\":" << signature.getDenominator() << "}"
        << ",\"repeat\":{\"start\":" << jsonBool(masterBar.hasRepeatStart())
        << ",\"end\":" << jsonBool(masterBar.hasRepeatEnd())
        << ",\"count\":" << masterBar.repeatCount()
        << ",\"alternate_ending\":" << jsonBool(masterBar.hasAlternateEndings())
        << ",\"alternate_ending_mask\":"
        << static_cast<std::uint32_t>(masterBar.alternateEndingMask())
        << "}"
        << ",\"double_bar\":" << jsonBool(masterBar.hasDoubleBar())
        << ",\"free_time\":" << jsonBool(masterBar.hasFreeTime())
        << ",\"anacrusis\":" << jsonBool(masterBar.hasAnacrusis())
        << ",\"fermata_present\":" << jsonBool(masterBar.hasFermata())
        << ",\"fermatas\":[";
    bool firstFermata = true;
    for (const auto& fermata : masterBar.fermatas()) {
        if (!firstFermata) out << ",";
        firstFermata = false;
        out << "{\"type_code\":" << enumCode(fermata.type)
            << ",\"offset\":";
        writeRational(out, fermata.offset);
        out << ",\"length\":" << fermata.length << "}";
    }
    const bool extendedAlternateEndings =
        masterBar.hasExtendedAlternateEndings();
    out << "]"
        << ",\"extended_alternate_endings_present\":"
        << jsonBool(extendedAlternateEndings)
        << ",\"extended_alternate_endings\":[";
    bool firstExtendedEnding = true;
    if (extendedAlternateEndings) {
        for (unsigned int endingIndex = 1; endingIndex <= 32; ++endingIndex) {
            if (!masterBar.isExtendedAlternateEndingSet(endingIndex)) continue;
            if (!firstExtendedEnding) out << ",";
            firstExtendedEnding = false;
            out << endingIndex;
        }
    }
    out << "]"
        << ",\"key_accidental_count\":"
        << masterBar.concertKeySignature().accidentalCount()
        << ",\"section_present\":" << jsonBool(sectionPresent)
        << ",\"section_letter\":";
    if (sectionPresent) {
        out << "\"" << escapeJson(masterBar.section().letter) << "\"";
    } else {
        out << "null";
    }
    out << ",\"section_text\":";
    if (sectionPresent) {
        out << "\"" << escapeJson(masterBar.section().text) << "\"";
    } else {
        out << "null";
    }
    out << ",\"triplet_feel_present\":" << jsonBool(tripletFeelPresent);
    writeOptionalNamedEnumFields(
        out,
        "triplet_feel",
        tripletFeelPresent,
        [&]() { return masterBar.tripletFeel(); },
        [](gp::core::MasterBar::TripletFeel value) {
            return utf8(gp::core::MasterBar::tripletFeelToQString(value));
        });
    out << ",\"directions_present\":" << jsonBool(!directions.empty())
        << ",\"directions\":[";
    bool firstDirection = true;
    for (const auto direction : directions) {
        if (!firstDirection) out << ",";
        firstDirection = false;
        out << "{\"code\":" << enumCode(direction) << ",\"name\":";
        writeOptionalOfficialName(
            out,
            [&]() {
                return utf8(gp::core::MasterTrack::directionToQString(direction));
            });
        out << "}";
    }
    out << "]";
}

void writeBar(
    std::ostream& out,
    const gp::core::Bar& bar,
    const gp::core::Staff& staff,
    const BeatSemanticRefs& beatRefs,
    const NoteSemanticRefs& noteRefs,
    unsigned int trackIndex,
    unsigned int sourceTrackIndex,
    unsigned int staffIndex) {
    const auto simile = bar.simileMark();
    const auto ottavia = bar.ottavia();
    const bool similePresent = enumCode(simile) != 0;
    const bool ottaviaPresent = enumCode(ottavia) != 0;
    out << "{\"semantic_id\":\""
        << escapeJson(measureSemanticId(trackIndex, staffIndex, bar.index()))
        << "\""
        << ",\"measure_index\":" << bar.index()
        << ",\"staff_index\":" << bar.staffIndex()
        << ",\"simile_present\":" << jsonBool(similePresent);
    writeOptionalNamedEnumFields(
        out,
        "simile",
        similePresent,
        [&]() { return simile; },
        [](gp::core::SimileMark value) {
            return gp::core::simileMarkToString(value);
        });
    out << ",\"ottavia_present\":" << jsonBool(ottaviaPresent);
    writeOptionalNamedEnumFields(
        out,
        "ottavia",
        ottaviaPresent,
        [&]() { return ottavia; },
        [](gp::core::Ottavia value) {
            return gp::core::ottaviaToString(value);
        });
    const auto masterBar = bar.masterBar();
    if (!masterBar) {
        throw std::runtime_error("selected measure has no MasterBar");
    }
    const auto drawingKeySignature =
        masterBar->drawingKeySignatureOfTrackAndStaff(
            sourceTrackIndex,
            staffIndex);
    out << ",\"master_measure_index\":" << masterBar->index()
        << ",\"drawing_key_signature\":{\"accidental_count\":"
        << drawingKeySignature.accidentalCount() << "}"
        << ",\"voices\":[";
    for (unsigned int voiceIndex = 0; voiceIndex < bar.voiceCount(); ++voiceIndex) {
        if (voiceIndex) out << ",";
        const auto voice = bar.voice(voiceIndex);
        if (!voice) {
            out << "null";
            continue;
        }
        out << "{\"semantic_id\":\""
            << escapeJson(voiceSemanticId(
                trackIndex,
                staffIndex,
                bar.index(),
                voice->index()))
            << "\""
            << ",\"voice_index\":" << voice->index()
            << ",\"events\":[";
        for (unsigned int beatIndex = 0; beatIndex < voice->beatCount(); ++beatIndex) {
            if (beatIndex) out << ",";
            const auto beat = voice->beat(beatIndex);
            if (beat) {
                writeBeat(
                    out,
                    *beat,
                    staff,
                    beatRefs,
                    noteRefs,
                    trackIndex,
                    staffIndex,
                    bar.index(),
                    voice->index());
            } else {
                out << "null";
            }
        }
        out << "]}";
    }
    out << "]}";
}

void writeStaff(
    std::ostream& out,
    const gp::core::Staff& staff,
    const BeatSemanticRefs& beatRefs,
    const NoteSemanticRefs& noteRefs,
    unsigned int trackIndex,
    unsigned int sourceTrackIndex) {
    const auto& tuning = staff.tuning();
    const unsigned int stringCount = tuning.stringCount();
    bool partialCapoPresent = false;
    for (unsigned int stringIndex = 0; stringIndex < stringCount; ++stringIndex) {
        if (staff.hasPartialCapoOnString(stringIndex)) {
            partialCapoPresent = true;
            break;
        }
    }
    out << "{\"semantic_id\":\""
        << escapeJson(staffSemanticId(trackIndex, staff.index())) << "\""
        << ",\"staff_index\":" << staff.index()
        << ",\"string_count\":" << stringCount
        << ",\"capo_present\":" << jsonBool(staff.capoFret() != 0)
        << ",\"capo_fret\":" << static_cast<unsigned int>(staff.capoFret())
        << ",\"partial_capo_present\":" << jsonBool(partialCapoPresent)
        << ",\"partial_capo\":[";
    bool firstPartialCapo = true;
    for (unsigned int stringIndex = 0; stringIndex < stringCount; ++stringIndex) {
        if (!staff.hasPartialCapoOnString(stringIndex)) continue;
        if (!firstPartialCapo) out << ",";
        firstPartialCapo = false;
        out << "{\"native_string_index\":" << stringIndex
            << ",\"fret\":" << static_cast<unsigned int>(
                staff.partialCapoFretOnString(stringIndex)) << "}";
    }
    out << "]"
        << ",\"tuning_pitches_low_to_high\":[";
    for (unsigned int stringIndex = 0; stringIndex < stringCount; ++stringIndex) {
        if (stringIndex) out << ",";
        out << tuning.tuning(stringIndex);
    }
    out << "]"
        << ",\"open_string_frets_low_to_high\":[";
    for (unsigned int stringIndex = 0; stringIndex < stringCount; ++stringIndex) {
        if (stringIndex) out << ",";
        out << staff.openStringFret(stringIndex);
    }
    out << "]"
        << ",\"total_capo_frets_low_to_high\":[";
    for (unsigned int stringIndex = 0; stringIndex < stringCount; ++stringIndex) {
        if (stringIndex) out << ",";
        out << static_cast<unsigned int>(
            staff.totalCapoFretOnString(stringIndex));
    }
    out << "]"
        << ",\"tuning_displayed_label\":\""
        << escapeJson(tuning.displayedLabel()) << "\""
        << ",\"tuning_label_visible\":"
        << jsonBool(tuning.labelVisible())
        << ",\"tuning_role_code\":" << enumCode(tuning.role())
        << ",\"tuning_is_flat\":" << jsonBool(tuning.isFlat())
        << ",\"short_drone_string\":"
        << jsonBool(tuning.hasShortDroneString())
        << ",\"diagram_collection\":";
    writeDiagramCollection(out, staff);
    out << ",\"measures\":[";
    for (unsigned int barIndex = 0; barIndex < staff.barCount(); ++barIndex) {
        if (barIndex) out << ",";
        const auto bar = staff.bar(barIndex);
        if (bar) {
            writeBar(
                out,
                *bar,
                staff,
                beatRefs,
                noteRefs,
                trackIndex,
                sourceTrackIndex,
                staff.index());
        } else {
            out << "null";
        }
    }
    out << "]}";
}

void writeTrack(
    std::ostream& out,
    const gp::core::Track& track,
    unsigned int trackIndex,
    unsigned int sourceTrackIndex) {
    const SemanticRefs refs = collectSemanticRefs(track, trackIndex);
    out << "{\"semantic_id\":\"" << escapeJson(trackSemanticId(trackIndex))
        << "\""
        << ",\"track_index\":" << trackIndex
        << ",\"name\":\"" << escapeJson(track.name()) << "\""
        << ",\"short_name\":\"" << escapeJson(track.shortName()) << "\""
        << ",\"let_ring_throughout\":" << jsonBool(track.hasLetRingThroughout())
        << ",\"staves\":[";
    for (unsigned int staffIndex = 0; staffIndex < track.staffCount(); ++staffIndex) {
        if (staffIndex) out << ",";
        const auto staff = track.staff(staffIndex);
        if (staff) {
            writeStaff(
                out,
                *staff,
                refs.beats,
                refs.notes,
                trackIndex,
                sourceTrackIndex);
        } else {
            out << "null";
        }
    }
    out << "]}";
}

} // namespace

bool writeScoreDump(
    const gp::core::Score& score,
    const gp::core::style::Stylesheet& stylesheet,
    const std::string& outputPath,
    std::string& error,
    int selectedTrackIndex) {
    if (outputPath.empty()) return true;
    if (selectedTrackIndex < 0
            || selectedTrackIndex >= static_cast<int>(score.trackCount())) {
        error = "selected source track index is out of range";
        return false;
    }
    const auto selectedTrack = score.track(
        static_cast<unsigned int>(selectedTrackIndex));
    if (!selectedTrack) {
        error = "selected source track is unavailable";
        return false;
    }
    if (selectedTrack->staffCount() == 0) {
        error = "selected source track has no staff";
        return false;
    }
    const auto masterTrack = score.masterTrack();
    if (!masterTrack) {
        error = "official score has no MasterTrack";
        return false;
    }
    if (masterTrack->masterBarCount() != selectedTrack->barCount()) {
        error = "MasterTrack measure count differs from selected track bar count";
        return false;
    }
    for (unsigned int masterMeasureIndex = 0;
         masterMeasureIndex < masterTrack->masterBarCount();
         ++masterMeasureIndex) {
        const auto masterBar = masterTrack->masterBar(masterMeasureIndex);
        if (!masterBar || masterBar->index() != masterMeasureIndex) {
            error = "MasterTrack contains an invalid MasterBar identity";
            return false;
        }
    }
    for (unsigned int staffIndex = 0;
         staffIndex < selectedTrack->staffCount();
         ++staffIndex) {
        const auto staff = selectedTrack->staff(staffIndex);
        if (!staff || staff->index() != staffIndex) {
            error = "selected track contains an invalid staff identity";
            return false;
        }
        if (staff->barCount() != selectedTrack->barCount()) {
            error = "selected staff measure count differs from selected track bar count";
            return false;
        }
        for (unsigned int barIndex = 0;
             barIndex < staff->barCount();
             ++barIndex) {
            const auto bar = staff->bar(barIndex);
            if (!bar
                    || bar->index() != barIndex
                    || bar->staffIndex() != staffIndex) {
                error = "selected staff contains an invalid measure identity";
                return false;
            }
            const auto masterBar = bar->masterBar();
            if (!masterBar || masterBar->index() != barIndex) {
                error = "selected measure has no matching MasterBar identity";
                return false;
            }
        }
    }

    const QString qOutput = QString::fromUtf8(outputPath.c_str());
    QDir().mkpath(QFileInfo(qOutput).absolutePath());
    const std::string temporaryPath = outputPath + ".tmp";
    QFile::remove(QString::fromUtf8(temporaryPath.c_str()));

    try {
        std::ofstream out(temporaryPath, std::ios::binary);
        if (!out) {
            error = "failed to open official score output";
            return false;
        }
        out << std::setprecision(12);
        out << "{\"schema\":\"gpomr.official-score\""
            << ",\"pdf_ok\":true"
            << ",\"source_track_index\":" << selectedTrackIndex;

        out << ",\"document\":{\"master_measures\":[";
        for (unsigned int masterMeasureIndex = 0;
             masterMeasureIndex < masterTrack->masterBarCount();
             ++masterMeasureIndex) {
            if (masterMeasureIndex) out << ",";
            const auto masterBar = masterTrack->masterBar(masterMeasureIndex);
            out << "{";
            writeMasterBar(out, *masterBar, *masterTrack);
            out << "}";
        }
        out << "],\"metadata\":";
        writeScoreMetadata(out, score, stylesheet);
        out << ",\"diagram_style\":";
        writeDiagramStyle(out, stylesheet);
        out
            << ",\"tempo\":{\"value\":" << masterTrack->tempoValue()
            << ",\"visible\":" << jsonBool(masterTrack->tempoVisible())
            << ",\"label\":\"" << escapeJson(masterTrack->tempoLabel()) << "\""
            << ",\"unit_code\":" << enumCode(masterTrack->tempoUnit())
            << ",\"unit_name\":";
        writeOptionalOfficialName(
            out,
            [&]() { return gp::core::tempoUnitToString(masterTrack->tempoUnit()); });
        out << "},\"tempo_automations\":";
        writeTempoAutomations(out, *masterTrack);
        out << "}";

        out << ",\"tracks\":[";
        writeTrack(
            out,
            *selectedTrack,
            0,
            static_cast<unsigned int>(selectedTrackIndex));
        out << "]}\n";
        out.close();
        if (!out) {
            error = "failed while writing official score output";
            QFile::remove(QString::fromUtf8(temporaryPath.c_str()));
            return false;
        }

        QFile::remove(qOutput);
        if (!QFile::rename(QString::fromUtf8(temporaryPath.c_str()), qOutput)) {
            error = "failed to write official score output";
            QFile::remove(QString::fromUtf8(temporaryPath.c_str()));
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = std::string("official score export failed: ") + exception.what();
    } catch (...) {
        error = "official score export failed with an unknown exception";
    }
    QFile::remove(QString::fromUtf8(temporaryPath.c_str()));
    return false;
}

} // namespace gpomr
