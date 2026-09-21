"""市販の楽譜ソフトが出す MusicXML を読めるかのテスト。

同梱の examples/*.musicxml は手書きの単一パート・タイ無しなので、ここでは実ソフトが
出す形（.mxl 圧縮・伴奏パート付き・タイ・繰り返し）を組み立てて確かめる。
"""
import zipfile

import pytest

from svs.score_io import parse_musicxml

_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         '<score-partwise version="4.0">\n')


def _note(step, octave, dur, lyric=None, tie=None):
    s = f'<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>' \
        f'<duration>{dur}</duration>'
    if tie:
        s += f'<tie type="{tie}"/>'
    if lyric:
        s += f'<lyric><syllabic>single</syllabic><text>{lyric}</text></lyric>'
    return s + '</note>'


def _score(parts_xml):
    ids = ''.join(f'<score-part id="{i}"><part-name>{n}</part-name></score-part>'
                  for i, n, _ in parts_xml)
    body = ''.join(f'<part id="{i}">{b}</part>' for i, _, b in parts_xml)
    return _HEAD + f'<part-list>{ids}</part-list>' + body + '</score-partwise>'


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return p


def test_mxl_matches_plain_xml(tmp_path):
    """.mxl（MuseScore の既定の保存形式）を開ける。中身は非圧縮版と同じ。"""
    xml = _score([('P1', 'Voice',
                   '<measure number="1"><attributes><divisions>480</divisions></attributes>'
                   '<direction><sound tempo="96"/></direction>'
                   + _note('C', 4, 480, 'あ') + _note('D', 4, 480, 'い')
                   + '</measure>')])
    plain = _write(tmp_path, 'a.musicxml', xml)
    mxl = tmp_path / 'a.mxl'
    with zipfile.ZipFile(mxl, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('META-INF/container.xml',
                   '<?xml version="1.0"?><container><rootfiles>'
                   '<rootfile full-path="a.musicxml"/></rootfiles></container>')
        z.write(plain, 'a.musicxml')
    assert parse_musicxml(mxl) == parse_musicxml(plain)


def test_mxl_without_container(tmp_path):
    """container.xml が無い .mxl でも、中の XML を拾って読む。"""
    xml = _score([('P1', 'Voice',
                   '<measure number="1"><attributes><divisions>1</divisions></attributes>'
                   + _note('C', 4, 1, 'あ') + '</measure>')])
    mxl = tmp_path / 'b.mxl'
    with zipfile.ZipFile(mxl, 'w') as z:
        z.writestr('b.musicxml', xml)
    notes, _ = parse_musicxml(mxl)
    assert [n.midi for n in notes] == [60]


def test_accompaniment_part_is_not_sung(tmp_path):
    """伴奏パートのノートを歌わない。既定は歌詞のある最初のパート。"""
    voice = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
             '<direction><sound tempo="90"/></direction>'
             + _note('C', 4, 1, 'あ') + _note('D', 4, 1, 'い') + '</measure>')
    piano = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
             + _note('C', 2, 2) + '</measure>')
    # 伴奏が先に書かれていても歌詞のある方を選ぶ
    p = _write(tmp_path, 'c.musicxml', _score([('P1', 'Piano', piano),
                                               ('P2', 'Voice', voice)]))
    notes, bpm = parse_musicxml(p)
    assert [n.midi for n in notes] == [60, 62]
    assert bpm == 90.0
    # 明示指定もできる（id でも part-name でも）
    assert [n.midi for n in parse_musicxml(p, part='P1')[0]] == [36]
    assert [n.midi for n in parse_musicxml(p, part='Piano')[0]] == [36]
    with pytest.raises(ValueError):
        parse_musicxml(p, part='Bass')


def test_tie_is_merged(tmp_path):
    """タイは 1 ノートに畳む。畳まないと伸ばしの途中で音素が切り直される。"""
    body = ('<measure number="1"><attributes><divisions>480</divisions></attributes>'
            '<direction><sound tempo="100"/></direction>'
            + _note('C', 4, 480, 'あ', tie='start')
            + '</measure><measure number="2">'
            + _note('C', 4, 960, tie='stop')
            + _note('D', 4, 480, 'い') + '</measure>')
    p = _write(tmp_path, 'd.musicxml', _score([('P1', 'Voice', body)]))
    notes, _ = parse_musicxml(p)
    assert [(n.midi, n.dur_beats, n.lyric) for n in notes] == [(60, 3.0, 'あ'), (62, 1.0, 'い')]


def test_tie_across_rest_is_not_merged(tmp_path):
    """休符を挟んだら繋がない（壊れた譜面で長さが暴走しないように）。"""
    body = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
            '<direction><sound tempo="100"/></direction>'
            + _note('C', 4, 1, 'あ', tie='start')
            + '<note><rest/><duration>1</duration></note>'
            + _note('C', 4, 1, tie='stop') + '</measure>')
    p = _write(tmp_path, 'e.musicxml', _score([('P1', 'Voice', body)]))
    notes, _ = parse_musicxml(p)
    assert [n.dur_beats for n in notes] == [1.0, 1.0, 1.0]


def test_tempo_from_metronome_when_no_sound_tag(tmp_path):
    """<sound tempo> が無くても <metronome> から読む（120 に落とさない）。"""
    body = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
            '<direction><direction-type><metronome><beat-unit>quarter</beat-unit>'
            '<per-minute>140</per-minute></metronome></direction-type></direction>'
            + _note('C', 4, 1, 'あ') + '</measure>')
    p = _write(tmp_path, 'f.musicxml', _score([('P1', 'Voice', body)]))
    assert parse_musicxml(p)[1] == 140.0


@pytest.mark.parametrize('frag, word', [
    ('<barline location="right"><repeat direction="backward"/></barline>', '繰り返し'),
    ('<backup><duration>1</duration></backup>', 'backup'),
    ('<direction><sound tempo="180"/></direction>', 'テンポ'),
])
def test_unsupported_notation_warns(tmp_path, capsys, frag, word):
    """未対応の記法は黙って違う音を出さず、警告を出す。"""
    body = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
            '<direction><sound tempo="90"/></direction>'
            + _note('C', 4, 1, 'あ') + frag + _note('D', 4, 1, 'い') + '</measure>')
    p = _write(tmp_path, 'g.musicxml', _score([('P1', 'Voice', body)]))
    parse_musicxml(p)
    assert word in capsys.readouterr().out


# ── 歌詞欄に音素を直接書く（英語のようにモーラ表を持たない言語向け）────────────

def _en_score(lyrics):
    body = '<measure number="1"><attributes><divisions>1</divisions></attributes>' \
           '<direction><sound tempo="90"/></direction>'
    for i, ly in enumerate(lyrics):
        body += _note('C', 4 + (i % 2), 1, ly)
    return _score([('P1', 'Voice', body + '</measure>')])


def test_phoneme_lyrics_need_no_mora_table(tmp_path):
    """英語は汎用のモーラ表を持たない。歌詞欄に ARPAbet を書けば表なしで読める。"""
    from svs.score_io import load_score
    p = _write(tmp_path, 'en.musicxml', _en_score(['k l ow s', 't uw', 'y uw']))
    got = [n.phonemes for ph in load_score(p, lang='en') for n in ph.notes]
    assert got == [['k', 'l', 'ow', 's'], ['t', 'uw'], ['y', 'uw']]


def test_english_word_still_needs_a_table(tmp_path):
    """単語はそのままでは読めない。音素表に無いので従来どおりモーラ表を引きに行く。"""
    from svs.score_io import load_score
    p = _write(tmp_path, 'en2.musicxml', _en_score(['close', 'to', 'you']))
    with pytest.raises(KeyError):
        load_score(p, lang='en')


def test_phoneme_lyrics_do_not_disturb_japanese(tmp_path):
    """日本語はモーラのまま読め、同じ譜面で音素直書きと混ぜられる。"""
    from svs.score_io import load_score
    body = ('<measure number="1"><attributes><divisions>1</divisions></attributes>'
            '<direction><sound tempo="120"/></direction>'
            + _note('C', 4, 1, 'か') + _note('D', 4, 1, 'ky o') + '</measure>')
    p = _write(tmp_path, 'ja.musicxml', _score([('P1', 'Voice', body)]))
    got = [n.phonemes for ph in load_score(p, lang='ja') for n in ph.notes]
    assert got == [['k', 'a'], ['ky', 'o']]
