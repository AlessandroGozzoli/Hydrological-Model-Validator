import pytest
import tempfile
import os

from reportlab.platypus import Paragraph, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from unittest.mock import Mock, patch
from reportlab.platypus import Table
from reportlab.pdfgen.canvas import Canvas
from io import BytesIO
from reportlab.platypus import Image as RLImage
from PIL import Image as PILImage


from Hydrological_model_validator.Report.report_utils import (MyDocTemplate,
                                                              PDFReportBuilder,
                                                              PositionedTable,
                                                              RotatedImage)

###############################################################################
# MyDocTemplate class tests
###############################################################################

# Dummy flowable to simulate flowables passed to afterFlowable
class DummyFlowable:
    def __init__(self, outlineLevel=None, text=""):
        self.outlineLevel = outlineLevel
        self._text = text
    def getPlainText(self):
        return self._text

# Test that __init__ creates the _headings attribute as an empty list
def test_init_creates_headings_list():
    doc = MyDocTemplate("file.pdf")
    # _headings should exist and be an empty list on init
    assert hasattr(doc, "_headings")
    assert isinstance(doc._headings, list)
    assert doc._headings == []

# Test afterFlowable calls notify with correct arguments when flowable has valid outlineLevel
def test_afterFlowable_calls_notify(monkeypatch):
    doc = MyDocTemplate("file.pdf")
    doc.page = 3  # simulate current page
    flowable = DummyFlowable(outlineLevel=1, text="Heading 1")

    called = {}
    # Patch notify method to capture call arguments
    def fake_notify(event, args):
        called['event'] = event
        called['args'] = args
    monkeypatch.setattr(doc, "notify", fake_notify)

    doc.afterFlowable(flowable)

    # Verify notify was called once with event 'TOCEntry'
    assert called.get('event') == 'TOCEntry'
    level, text, page = called.get('args')
    # Check that level, text, page match expected values
    assert level == 1
    assert text == "Heading 1"
    assert page == 3

# Test afterFlowable does not call notify if flowable has no outlineLevel attribute
def test_afterFlowable_no_outlineLevel(monkeypatch):
    doc = MyDocTemplate("file.pdf")
    doc.page = 5
    # Flowable without outlineLevel attribute
    class NoOutlineFlowable:
        def getPlainText(self):
            return "No Outline"
    flowable = NoOutlineFlowable()

    called = {'not_called': True}
    def fake_notify(event, args):
        called['not_called'] = False  # If notify called, flip flag
    monkeypatch.setattr(doc, "notify", fake_notify)

    doc.afterFlowable(flowable)

    # Notify should not have been called
    assert called['not_called']

# Test afterFlowable does not call notify if flowable.outlineLevel is None
def test_afterFlowable_outlineLevel_none(monkeypatch):
    doc = MyDocTemplate("file.pdf")
    doc.page = 7
    flowable = DummyFlowable(outlineLevel=None, text="No Level")

    called = {'not_called': True}
    def fake_notify(event, args):
        called['not_called'] = False
    monkeypatch.setattr(doc, "notify", fake_notify)

    doc.afterFlowable(flowable)

    # Notify should not be called for None outlineLevel
    assert called['not_called']

# Test afterFlowable passes exact flowable text and current page number to notify
def test_afterFlowable_text_and_page(monkeypatch):
    doc = MyDocTemplate("file.pdf")
    doc.page = 10
    flowable = DummyFlowable(outlineLevel=0, text="Main Heading")

    captured = {}
    def fake_notify(event, args):
        captured['event'] = event
        captured['args'] = args
    monkeypatch.setattr(doc, "notify", fake_notify)

    doc.afterFlowable(flowable)

    # Confirm that event is 'TOCEntry'
    assert captured['event'] == 'TOCEntry'
    level, text, page = captured['args']
    # Check all values exactly match what we expect
    assert level == 0
    assert text == "Main Heading"
    assert page == 10

###############################################################################
# PDFReportBuilder class tests
###############################################################################

# Test that PDFReportBuilder.__init__ sets key attributes and styles correctly
def test_init_creates_attributes_and_styles():
    pdf_path = "dummy.pdf"
    builder = PDFReportBuilder(pdf_path)

    # pdf_path attribute stores output file path
    assert builder.pdf_path == pdf_path

    # styles is a StyleSheet1 instance containing default and custom styles
    assert hasattr(builder, "styles")
    assert isinstance(builder.styles['Heading1'], ParagraphStyle)

    # Heading1 and Heading2 are centered
    assert builder.styles['Heading1'].alignment == TA_CENTER
    assert builder.styles['Heading2'].alignment == TA_CENTER

    # Custom styles 'CenteredTitle' and 'SectionTitle' added
    assert 'CenteredTitle' in builder.styles
    assert 'SectionTitle' in builder.styles

    # toc attribute exists and has levelStyles configured
    assert hasattr(builder, "toc")
    assert len(builder.toc.levelStyles) == 2
    # LevelStyles font sizes correspond to expected TOC style
    assert builder.toc.levelStyles[0].fontSize == 14
    assert builder.toc.levelStyles[1].fontSize == 12

# Build_title_page clears story and appends Paragraphs and PageBreak
def test_build_title_page_basic(monkeypatch):
    builder = PDFReportBuilder("dummy.pdf")

    # Patch print to catch logo warning if any
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    builder.build_title_page(
        title="Test Title",
        subtitle="Test Subtitle",
        author="Author Name",
        organization="Org",
        email="email@example.com",
        cellphone="12345",
        logo_path=None,
        package_name="nonexistent_package_for_version"  # to trigger fallback
    )

    # Story should be reset and contain multiple flowables including Paragraphs, Spacers, PageBreak
    assert len(builder.story) > 5

    # Last flowable should be PageBreak to separate title page from next content
    assert isinstance(builder.story[-1], PageBreak)

    # Check some Paragraphs contain the title and author
    texts = [f.getPlainText() for f in builder.story if isinstance(f, Paragraph)]
    assert any("Test Title" in t for t in texts)
    assert any("Author" in t for t in texts)

# Build_toc appends Paragraph, Flowable line, TOC, and PageBreak to story
def test_build_toc_appends_correct_flowables():
    builder = PDFReportBuilder("dummy.pdf")

    builder.build_toc()

    # Should have at least 4 flowables: title Paragraph, HorizontalLine, Spacer, toc, PageBreak
    assert len(builder.story) >= 4

    # First flowable is Paragraph with TOC title text
    assert isinstance(builder.story[0], Paragraph)
    assert "Table of Contents" in builder.story[0].getPlainText()

    # Last flowable is PageBreak to separate TOC page
    assert isinstance(builder.story[-1], PageBreak)

    # The TOC flowable is in the story
    assert builder.toc in builder.story

# Add_heading adds Paragraph with correct style, outlineLevel, and bookmark
def test_add_heading_creates_paragraph_with_metadata():
    builder = PDFReportBuilder("dummy.pdf")

    # Add level 0 heading
    builder.add_heading("Main Heading", level=0)
    para0 = builder.story[-1]
    assert isinstance(para0, Paragraph)
    assert para0.getPlainText() == "Main Heading"
    assert getattr(para0, "outlineLevel") == 0
    assert getattr(para0, "bookmark").startswith("heading_")

    # Add level 1 heading
    builder.add_heading("Sub Heading", level=1)
    para1 = builder.story[-1]
    assert isinstance(para1, Paragraph)
    assert para1.getPlainText() == "Sub Heading"
    assert getattr(para1, "outlineLevel") == 1
    assert getattr(para1, "bookmark").startswith("heading_")

    # Check that the style applied matches Heading1 or Heading2
    style_names = [para0.style.name, para1.style.name]
    assert "Heading1" in style_names[0]
    assert "Heading2" in style_names[1]

# Test save calls MyDocTemplate.multiBuild with the story content
def test_save_calls_multiBuild(monkeypatch):
    builder = PDFReportBuilder("dummy.pdf")

    # Add one heading to have some content
    builder.add_heading("Save Test")

    called_args = {}
    def fake_multiBuild(story):
        # Capture the passed story
        called_args['story'] = story
    # Patch MyDocTemplate to return an object with fake multiBuild
    class DummyDoc:
        def multiBuild(self, story):
            fake_multiBuild(story)
    monkeypatch.setattr("Hydrological_model_validator.Report.report_utils.MyDocTemplate", lambda path, pagesize=None: DummyDoc())

    builder.save()

    # Ensure multiBuild was called with builder.story as argument
    assert 'story' in called_args
    assert called_args['story'] == builder.story
    

###############################################################################
# PositionedTable class tests
###############################################################################


# Create a sample flowable to wrap in PositionedTable
@pytest.fixture
def sample_table():
    data = [['A', 'B'], ['C', 'D']]
    return Table(data)

# Constructor correctly sets attributes
def test_initialization_sets_attributes(sample_table):
    # Setup offsets
    x_offset, y_offset = 100, 200
    pt = PositionedTable(sample_table, x_offset, y_offset)

    # Verify values are stored correctly
    assert pt.flowable is sample_table
    assert pt.x_offset == x_offset
    assert pt.y_offset == y_offset

# Wrap delegates to the wrapped flowable's wrap method
def test_wrap_delegates_to_flowable(sample_table):
    # Mock the wrap method to return known dimensions
    sample_table.wrap = Mock(return_value=(300, 150))

    pt = PositionedTable(sample_table, 10, 20)

    # Call wrap with test available width/height
    result = pt.wrap(500, 700)

    # Ensure the mock was called with correct values
    sample_table.wrap.assert_called_once_with(500, 700)
    assert result == (300, 150)  # Should match what the mock returns

# Draw applies canvas transformation and draws the flowable
def test_draw_calls_canvas_correctly(sample_table):
    pt = PositionedTable(sample_table, 50, 75)

    # Set up mocks on canvas and the flowable
    mock_canvas = Mock()
    sample_table.drawOn = Mock()

    pt.canv = mock_canvas  # Inject mocked canvas into the PositionedTable

    # Call draw and verify canvas state management and translation
    pt.draw()

    # Check canvas state methods were called in order
    assert mock_canvas.saveState.called, "saveState() should be called"
    assert mock_canvas.translate.called, "translate() should be called"
    assert mock_canvas.restoreState.called, "restoreState() should be called"

    mock_canvas.translate.assert_called_once_with(50, 75)
    sample_table.drawOn.assert_called_once_with(mock_canvas, 0, 0)

# Integration — draw on an actual canvas to ensure no exceptions
def test_draw_on_real_canvas(sample_table):
    # Ensure the PositionedTable can draw to a real ReportLab canvas without crashing
    pt = PositionedTable(sample_table, 10, 20)

    # Create an in-memory canvas
    buffer = BytesIO()
    real_canvas = Canvas(buffer, pagesize=letter)

    pt.canv = real_canvas
    pt.wrap(0, 0)

    # Should not raise
    try:
        pt.draw()
    except Exception as e:
        pytest.fail(f"Draw raised an unexpected exception: {e}")
   
###############################################################################
# RotatedImage class tests
###############################################################################
        
# Helper to create a temporary test image and return its path
@pytest.fixture
def temp_image():
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image = PILImage.new('RGB', (400, 200), color='blue')  # width=400, height=200
        image.save(tmp.name)
        yield tmp.name
    os.remove(tmp.name)

# Constructor correctly loads image and computes scaled dimensions
def test_initialization_sets_attributes_rotated(temp_image):
    # Use frame smaller than image to force scaling
    ri = RotatedImage(temp_image, frame_width=300, frame_height=400, margin=10, header_height=20)

    # Check original image size loaded correctly
    assert ri.orig_width == 400
    assert ri.orig_height == 200

    # Scaling should occur based on rotated fit (rotate 90° => width becomes height)
    max_w = 300 - 2 * 10              # frame_width - 2 * margin
    max_h = 400 - 2 * 10 - 20         # frame_height - 2 * margin - header
    scale_w = max_w / 200             # width available / image height
    scale_h = max_h / 400             # height available / image width
    expected_scale = min(scale_w, scale_h, 1.0)

    # Draw dimensions should be scaled unrotated width/height
    assert abs(ri.draw_width - 400 * expected_scale) < 1e-6
    assert abs(ri.draw_height - 200 * expected_scale) < 1e-6

# Wrap method returns correctly computed dimensions
def test_wrap_returns_expected_dimensions(temp_image):
    ri = RotatedImage(temp_image, margin=5, header_height=15)
    wrapped = ri.wrap(1000, 1000)

    # wrap returns image dimensions with margins and header
    expected_w = ri.draw_width + 2 * 5
    expected_h = ri.draw_height + 2 * 5 + 15
    assert abs(wrapped[0] - expected_w) < 1e-6
    assert abs(wrapped[1] - expected_h) < 1e-6

# Draw calls canvas operations in expected order and performs transforms
def test_draw_with_default_offset(temp_image):
    # Setup RotatedImage instance with default offsets
    ri = RotatedImage(temp_image, frame_width=500, frame_height=700)

    # Mock the canvas and RLImage instance
    mock_canvas = Mock()
    mock_rl_image = Mock(spec=RLImage)

    # Patch RLImage in your module so RotatedImage.draw() uses the mocked RLImage
    with patch('Hydrological_model_validator.Report.report_utils.RLImage', return_value=mock_rl_image):
        ri.canv = mock_canvas
        ri.draw()

    # Assert canvas state save/restore called
    assert mock_canvas.saveState.called
    assert mock_canvas.restoreState.called

    # Assert rotation and translation transformations applied
    assert mock_canvas.rotate.called
    # At least two translate calls expected (one for initial translation, one after rotation)
    assert mock_canvas.translate.call_count >= 2

    # Confirm the image drawOn is called once with correct args
    mock_rl_image.drawOn.assert_called_once_with(mock_canvas, 0, 0)

# Draw uses custom_offset_y if provided
def test_draw_with_custom_offset(temp_image):
    custom_y = 123.45
    frame_w, frame_h = 456, 636
    ri = RotatedImage(temp_image, frame_width=frame_w, frame_height=frame_h, custom_offset_y=custom_y)

    mock_canvas = Mock()
    mock_rl_image = Mock(spec=RLImage)

    with patch('Hydrological_model_validator.Report.report_utils.RLImage', return_value=mock_rl_image):
        ri.canv = mock_canvas
        ri.draw()

    # Calculate expected horizontal offset (centering the rotated image)
    expected_offset_x = (frame_w - ri.draw_height) / 2

    # Verify the translate call with the custom offset_y was made (x, -custom_y)
    mock_canvas.translate.assert_any_call(expected_offset_x, -custom_y)

# Integration Test: render a full image to a canvas (no exceptions)
def test_draw_on_real_canvas_rotated(temp_image):
    ri = RotatedImage(temp_image)

    # Draw to in-memory canvas to ensure it works
    buffer = BytesIO()
    canvas = Canvas(buffer)
    ri.canv = canvas

    try:
        ri.draw()
    except Exception as e:
        pytest.fail(f"RotatedImage.draw() raised an exception: {e}")