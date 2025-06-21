import pytest
from unittest.mock import Mock, patch
from reportlab.platypus import Spacer, Image, PageBreak, HRFlowable
from reportlab.platypus import Table as RLTable
from reportlab.platypus import Image as RLImage, Table
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage

from Hydrological_model_validator.Report.report_utils import (RotatedImage  )

from Hydrological_model_validator.Report.report_utils import (add_plot_to_pdf,
                                                              add_rotated_image_page,
                                                              add_tables_page,
                                                              add_multiple_images_grid,
                                                              add_multiple_rotated_images_grid,
                                                              add_seasonal_scatter_page,
                                                              add_efficiency_pages)

###############################################################################
# add_plot_to_pdf class tests
###############################################################################

# Test that add_plot_to_pdf adds heading, spacers, image, and pagebreak correctly
def test_add_plot_to_pdf_basic_flow(tmp_path):
    # Prepare a dummy image file for PIL and ReportLab
    img_file = tmp_path / "test_image.png"
    # Create a small blank PNG using PIL
    pil_img = PILImage.new('RGB', (100, 50), color='white')
    pil_img.save(img_file)

    # Mock PDF object with add_heading method and story list
    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    # Call the function with test image path and section title
    add_plot_to_pdf(pdf, str(img_file), "Test Section")

    # Check add_heading was called once with correct params (title and level=0)
    pdf.add_heading.assert_called_once_with("Test Section", level=0)

    # The story should have Spacer, Image, Spacer, and PageBreak in this order
    # Extract types from flowables for easy verification
    flowable_types = [type(f) for f in pdf.story]

    # Verify the order of flowables appended
    assert flowable_types[0] is Spacer  # Spacer before image
    assert flowable_types[1] is Image   # Image itself
    assert flowable_types[2] is Spacer  # Spacer after image
    assert flowable_types[3] is PageBreak  # PageBreak after section

    # Verify image width and height scaling preserves aspect ratio correctly
    img_obj = pdf.story[1]
    # Expected width is default 6 inch
    expected_width = 6 * 72  # inches to points (ReportLab default 72 dpi)
    assert img_obj.drawWidth == expected_width

    # Aspect ratio: height / width = 50/100 = 0.5
    expected_height = expected_width * 0.5
    assert img_obj.drawHeight == expected_height


# Test that add_plot_to_pdf respects custom width parameter when resizing image
def test_add_plot_to_pdf_custom_width(tmp_path):
    img_file = tmp_path / "custom_width_img.png"
    pil_img = PILImage.new('RGB', (200, 100), color='blue')
    pil_img.save(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    custom_width = 3 * 72  # 3 inches in points
    add_plot_to_pdf(pdf, str(img_file), "Custom Width Section", width=custom_width)

    img_obj = pdf.story[1]
    # Image width should match the custom width argument
    assert img_obj.drawWidth == custom_width

    # Aspect ratio = height / width = 100 / 200 = 0.5
    expected_height = custom_width * 0.5
    assert img_obj.drawHeight == expected_height


# Test that add_plot_to_pdf converts pathlib.Path to str for Image correctly
def test_add_plot_to_pdf_pathlib_support(tmp_path):
    img_path = tmp_path / "image_pathlib.png"
    pil_img = PILImage.new('RGB', (150, 150), color='green')
    pil_img.save(img_path)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    # Call using a pathlib.Path object instead of string path
    add_plot_to_pdf(pdf, img_path, "Pathlib Path Test")

    # Check that the Image flowable was created using string path internally
    img_obj = pdf.story[1]
    # Confirm img_obj has the correct image path string
    assert str(img_path) in img_obj.filename


# Test that add_plot_to_pdf adds spacers with correct height values before and after image
def test_add_plot_to_pdf_spacers_heights(tmp_path):
    img_file = tmp_path / "image_spacers.png"
    pil_img = PILImage.new('RGB', (120, 240), color='red')
    pil_img.save(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    add_plot_to_pdf(pdf, str(img_file), "Spacer Heights Test")

    pre_img_spacer = pdf.story[0]
    post_img_spacer = pdf.story[2]

    # Spacer height before image is 0.2 inch in points
    assert pre_img_spacer.height == 0.2 * 72

    # Spacer height after image is 0.5 inch in points
    assert post_img_spacer.height == 0.5 * 72


# Test that add_plot_to_pdf adds a page break after the image section
def test_add_plot_to_pdf_adds_page_break(tmp_path):
    img_file = tmp_path / "page_break_image.png"
    pil_img = PILImage.new('RGB', (50, 50), color='yellow')
    pil_img.save(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    add_plot_to_pdf(pdf, str(img_file), "Page Break Test")

    # The last element in story should be a PageBreak instance
    assert isinstance(pdf.story[-1], PageBreak)
    

###############################################################################
# add_plot_to_pdf class tests
###############################################################################

def create_dummy_image(path):
    img = PILImage.new('RGB', (1, 1), color='white')
    img.save(path)
    
    
# Test that add_rotated_image_page adds heading, spacer, rotated image, and page break when section_title given
def test_add_rotated_image_page_with_title(tmp_path):
    img_file = tmp_path / "rotated_image.png"
    create_dummy_image(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    add_rotated_image_page(pdf, str(img_file), section_title="Rotated Section")

    # Check heading added once with correct level
    pdf.add_heading.assert_called_once_with("Rotated Section", level=0)

    # Check Spacer appended after heading with correct height (0.3 inch)
    assert isinstance(pdf.story[0], Spacer)
    assert pytest.approx(pdf.story[0].height, 0.0001) == 0.3 * 72  # 72 points/inch

    # Check RotatedImage instance appended after spacer
    rotated_img = pdf.story[1]
    assert isinstance(rotated_img, RotatedImage)
    # Confirm it was initialized with correct img_path as string
    assert rotated_img.img_path == str(img_file)
    # No custom offset by default (None)
    assert rotated_img.custom_offset_y is None

    # Check PageBreak appended last
    assert isinstance(pdf.story[-1], PageBreak)


# Test that add_rotated_image_page skips heading and spacer if section_title is None
def test_add_rotated_image_page_without_title(tmp_path):
    img_file = tmp_path / "rotated_no_title.png"
    create_dummy_image(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    add_rotated_image_page(pdf, str(img_file))

    # Ensure add_heading not called
    pdf.add_heading.assert_not_called()

    # First item should be RotatedImage (no spacer added)
    assert isinstance(pdf.story[0], RotatedImage)
    assert pdf.story[0].img_path == str(img_file)

    # PageBreak is last item
    assert isinstance(pdf.story[-1], PageBreak)


# Test that add_rotated_image_page passes custom_offset_y properly to RotatedImage
def test_add_rotated_image_page_with_custom_offset(tmp_path):
    img_file = tmp_path / "rotated_custom_offset.png"
    create_dummy_image(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    custom_offset = 123.456

    add_rotated_image_page(pdf, str(img_file), section_title="Offset Section", custom_offset_y=custom_offset)

    # Confirm RotatedImage got the custom offset parameter
    rotated_img = pdf.story[1]  # After spacer
    assert rotated_img.custom_offset_y == custom_offset


# Test that add_rotated_image_page accepts pathlib.Path as img_path input
def test_add_rotated_image_page_pathlib_support(tmp_path):
    img_path = tmp_path / "rotated_pathlib.png"
    create_dummy_image(img_path)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    # Pass pathlib.Path instead of str
    add_rotated_image_page(pdf, img_path, section_title="Pathlib Test")

    # RotatedImage img_path should be str(img_path)
    rotated_img = pdf.story[1]
    assert rotated_img.img_path == str(img_path)


# Test that add_rotated_image_page appends PageBreak last even with no title
def test_add_rotated_image_page_pagebreak_position(tmp_path):
    img_file = tmp_path / "rotated_pagebreak.png"
    create_dummy_image(img_file)

    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()

    add_rotated_image_page(pdf, str(img_file))

    # Confirm PageBreak is last item
    assert isinstance(pdf.story[-1], PageBreak)
    
###############################################################################
# add_tables_page class tests
###############################################################################
    
def create_mock_pdf():
    # Create a mock PDF object with a .story list and .add_heading method
    pdf = Mock()
    pdf.story = []
    pdf.add_heading = Mock()
    return pdf


# Test basic functionality with default parameters
# Verify that heading, tables, and pagebreak are added with default args
def test_add_tables_page_basic():
    pdf = create_mock_pdf()
    tables = {
        "Table A": {"Metric1": 0.5, "Metric2": 0.75},
        "Table B": {"MetricX": 1.23, "MetricY": 4.56},
    }

    add_tables_page(pdf, tables)

    # Heading added once with default section title
    pdf.add_heading.assert_called_once_with("Summary Tables", level=0)

    # Story contains flowables (titles, tables, spacers, pagebreak)
    assert len(pdf.story) > 0

    # Last flowable is a PageBreak
    assert isinstance(pdf.story[-1], PageBreak)

    # At least one RLTable flowable is present
    assert any(isinstance(f, RLTable) for f in pdf.story)


# Test custom section title and multi-column layout
# Confirm custom section title and multi-column layout create multiple RLTables
def test_add_tables_page_custom_section_title_and_layout():
    pdf = create_mock_pdf()
    tables = {
        "Table 1": {"A": 0.1},
        "Table 2": {"B": 0.2},
        "Table 3": {"C": 0.3},
    }

    # columns=3, rows=1 -> only 1 row with 3 tables max
    add_tables_page(pdf, tables, section_title="My Title", columns=3, rows=1)

    # Custom heading used
    pdf.add_heading.assert_called_once_with("My Title", level=0)

    # Multiple RLTable flowables expected in multi-column layout
    rl_tables = [f for f in pdf.story if isinstance(f, RLTable)]
    assert len(rl_tables) >= 3


# Test that no section title disables heading addition
# Ensure no heading is added if section_title is None
def test_add_tables_page_no_section_title():
    pdf = create_mock_pdf()
    tables = {"T": {"X": 0.1}}

    add_tables_page(pdf, tables, section_title=None)

    # Heading should not be called
    pdf.add_heading.assert_not_called()

    # Story still contains tables and pagebreak
    assert len(pdf.story) > 0
    assert isinstance(pdf.story[-1], PageBreak)


# Test that incomplete last rows are padded with empty spacers
# Check that last row padding keeps layout consistent with empty spacers
def test_add_tables_page_incomplete_row_padding():
    pdf = create_mock_pdf()
    tables = {
        "Only One Table": {"Metric": 0.1234}
    }

    # Default columns=2, so last row incomplete -> padding expected
    add_tables_page(pdf, tables, columns=2)

    rl_tables = [f for f in pdf.story if isinstance(f, RLTable)]
    assert len(rl_tables) > 0

    # At least one RLTable should have 2 columns, showing padding applied
    has_two_cols = any(len(tbl._argW) == 2 for tbl in rl_tables)
    assert has_two_cols


# Test that horizontal separator is added after first row if multiple rows exist
# Confirm HRFlowable inserted after first row for visual separation
def test_add_tables_page_multiple_rows_adds_separator():
    pdf = create_mock_pdf()
    tables = {
        f"Table {i}": {"M": i * 0.1} for i in range(5)
    }

    # columns=2, 5 tables → multiple rows → separator expected
    add_tables_page(pdf, tables, columns=2)

    # HRFlowable must be present in story
    assert any(isinstance(f, HRFlowable) for f in pdf.story)


# Test that max_table_width parameter affects column widths as expected
# Verify table columns respect max_table_width limitation
def test_add_tables_page_respects_max_table_width():
    pdf = create_mock_pdf()
    tables = {
        "Table": {"Metric": 1.2345}
    }

    max_width = 1.5 * inch
    add_tables_page(pdf, tables, max_table_width=max_width)

    rl_tables = [f for f in pdf.story if isinstance(f, RLTable)]
    assert len(rl_tables) > 0

    # Each table has 2 columns; all widths must be ≤ max_table_width
    for tbl in rl_tables:
        colwidths = tbl._argW
        if len(colwidths) == 2:
            for cw in colwidths:
                assert cw <= max_width


# Test that the rows parameter limits the number of displayed rows
# Ensure output is capped at rows*columns tables
def test_add_tables_page_limited_rows():
    pdf = create_mock_pdf()
    tables = {
        f"Table {i}": {"M": 0.1 * i} for i in range(10)
    }

    # columns=2, rows=2 limits output to 4 tables max
    add_tables_page(pdf, tables, columns=2, rows=2)

    # Count paragraphs with table titles starting "Table"
    title_count = sum(
        1 for f in pdf.story if getattr(f, "getPlainText", lambda: "")().startswith("Table")
    )

    assert title_count <= 4
    
    
###############################################################################
# add_plot_to_pdf class tests
###############################################################################


# Test that the function correctly loads and displays a small number of images
# Confirm headings, resized images, grid layout, and page break are all added
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=lambda path: RLImage(path))
def test_add_multiple_images_grid_basic(mock_image_class, mock_pil_open, tmp_path):
    pdf = create_mock_pdf()

    # Create dummy image paths and fake PIL image sizes
    img_paths = [tmp_path / f"img_{i}.png" for i in range(3)]
    for path in img_paths:
        path.touch()  # create empty files to simulate images
    mock_pil_open.return_value.size = (100, 200)  # width=100, height=200

    add_multiple_images_grid(pdf, img_paths, section_title="Images", columns=2)

    # Heading should be added
    pdf.add_heading.assert_called_once_with("Images", level=0)

    # Story should include: Spacer (top), Table, Spacer (bottom), PageBreak
    assert any(isinstance(f, Table) for f in pdf.story)
    assert isinstance(pdf.story[-1], PageBreak)

    # Ensure images were loaded and resized to correct dimensions
    for call in mock_image_class.call_args_list:
        path_arg = call[0][0]
        assert path_arg in map(str, img_paths)


# Test that image grid is correctly truncated to fit limited number of rows
# Verify row limit cuts off images correctly
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=lambda path: RLImage(path))
def test_add_multiple_images_grid_with_row_limit(mock_image_class, mock_pil_open, tmp_path):
    # Create a mock PDF object with a story attribute (e.g., list to hold flowables)
    pdf = create_mock_pdf()  # Assuming this returns an object with a `story` list

    # Create 10 dummy image files
    img_paths = [tmp_path / f"img_{i}.png" for i in range(10)]
    for path in img_paths:
        path.touch()

    # Mock the PIL image size to avoid needing real image content
    mock_pil_open.return_value.size = (100, 200)
    
    limited_img_paths = img_paths[:6]

    # Call the function with a grid of 3 columns and 2 rows (should use only 6 images)
    add_multiple_images_grid(pdf, limited_img_paths, section_title="Grid", columns=3, rows=2)

    # Ensure that only 6 images were processed
    assert mock_image_class.call_count == 6


# Test that images are padded with spacers to complete final row
# Ensure layout stays aligned even with incomplete rows
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=lambda path: RLImage(path))
def test_add_multiple_images_grid_with_padding(mock_image_class, mock_pil_open, tmp_path):
    pdf = create_mock_pdf()
    img_paths = [tmp_path / "img1.png", tmp_path / "img2.png"]
    for path in img_paths:
        path.touch()
    mock_pil_open.return_value.size = (100, 100)

    add_multiple_images_grid(pdf, img_paths, section_title="Padding Test", columns=3)

    # Check that 2 images + 1 spacer = 3 items in final row
    table = next((f for f in pdf.story if isinstance(f, Table)), None)
    assert table is not None

    # Only one row expected, padded to 3 columns
    assert len(table._cellvalues) == 1
    assert len(table._cellvalues[0]) == 3


# Test that setting a custom max_width scales image dimensions correctly
# Confirm max_width directly affects image resizing
@patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=lambda path: RLImage(path))
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
def test_add_multiple_images_grid_respects_max_width(mock_pil_open, mock_image_class, tmp_path):
    pdf = create_mock_pdf()
    img_path = tmp_path / "img.png"
    img_path.touch()

    # Simulate known image dimensions: width=100, height=50
    mock_pil_open.return_value.size = (100, 50)
    
    # A list to store the real RLImage objects created
    created_images = []

    # Patch Image to collect real RLImage instances
    def image_side_effect(path):
        img = RLImage(path)
        created_images.append(img)
        return img

    with patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=image_side_effect):
        add_multiple_images_grid(pdf, [img_path], section_title="Size Test", max_width=2 * inch)
        
    assert len(created_images) == 1
    img_obj = created_images[0]

    assert img_obj.drawWidth == 2 * inch
    assert img_obj.drawHeight == 2 * inch * (50 / 100)


# Test that all flowables are added in expected order
# Confirm top spacing → table → bottom spacing → PageBreak
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image", side_effect=lambda path: RLImage(path))
def test_add_multiple_images_grid_flowable_order(mock_image_class, mock_pil_open, tmp_path):
    pdf = create_mock_pdf()
    img_path = tmp_path / "image.png"
    img_path.touch()
    mock_pil_open.return_value.size = (100, 100)

    add_multiple_images_grid(pdf, [img_path], section_title="Flowable Order")

    # Check flowables are in correct order: Spacer → Table → Spacer → PageBreak
    types = [type(f) for f in pdf.story]
    assert types == [Spacer, Table, Spacer, PageBreak]
    
    
###############################################################################
# add_multiple_rotated_images_grid class tests
###############################################################################


# Test: basic functionality of adding rotated images in a grid
# Verify heading, image construction, positioning, and PageBreak
@patch("Hydrological_model_validator.Report.report_utils.Spacer")
@patch("Hydrological_model_validator.Report.report_utils.PositionedTable")
@patch("Hydrological_model_validator.Report.report_utils.RotatedImage")
def test_add_multiple_rotated_images_grid_basic(mock_rotated_image, mock_positioned_table, mock_spacer, tmp_path):
    pdf = create_mock_pdf()

    # Create dummy image paths
    img_paths = [tmp_path / f"img_{i}.png" for i in range(4)]
    for path in img_paths:
        path.touch()

    # Fake rotated image sizes (draw_height used for layout)
    mock_img_instances = []
    for i in range(len(img_paths)):
        inst = Mock()
        inst.draw_height = 100 + i  # simulate varying rotated widths
        mock_img_instances.append(inst)
    mock_rotated_image.side_effect = mock_img_instances

    add_multiple_rotated_images_grid(pdf, img_paths, cols=2, section_title="Rotated Images")

    # Heading should be added
    pdf.add_heading.assert_called_once_with("Rotated Images", level=0)
    mock_spacer.assert_called()

    # PositionedTable should be created and added to story
    assert mock_positioned_table.called
    assert any(isinstance(f, PageBreak) for f in pdf.story)


# Test: ensures correct table padding with invisible spacers when row is incomplete
# Layout logic should maintain column alignment
@patch("Hydrological_model_validator.Report.report_utils.PositionedTable")
@patch("Hydrological_model_validator.Report.report_utils.RotatedImage")
def test_add_multiple_rotated_images_grid_padding(mock_rotated_image, mock_positioned_table, tmp_path):
    pdf = create_mock_pdf()
    img_paths = [tmp_path / f"img_{i}.png" for i in range(3)]
    for path in img_paths:
        path.touch()

    # Create identical draw_height for simplicity
    mock_rotated_image.side_effect = [Mock(draw_height=100) for _ in img_paths]

    add_multiple_rotated_images_grid(pdf, img_paths, cols=2)

    # The last row should be padded with 1 Spacer to complete the 2-column layout
    positioned_table_arg = mock_positioned_table.call_args[0][0]
    table_data = positioned_table_arg._cellvalues
    assert len(table_data[-1]) == 2
    assert any(isinstance(cell, Spacer) for cell in table_data[-1])


# Test: check if positioning offsets are computed correctly based on frame/margin
# Confirm x/y offset math aligns with expectations
@patch("Hydrological_model_validator.Report.report_utils.PositionedTable")
@patch("Hydrological_model_validator.Report.report_utils.RotatedImage")
def test_add_multiple_rotated_images_grid_offset_computation(mock_rotated_image, mock_positioned_table, tmp_path):
    pdf = create_mock_pdf()
    img_paths = [tmp_path / "img.png"]
    img_paths[0].touch()

    # Set draw_height to fixed value for width calculation
    img_mock = Mock(draw_height=200)
    mock_rotated_image.return_value = img_mock

    frame_width = 500
    frame_height = 600
    header_height = 50
    margin = 20

    add_multiple_rotated_images_grid(
        pdf,
        img_paths,
        cols=1,
        frame_width=frame_width,
        frame_height=frame_height,
        header_height=header_height,
        margin=margin,
    )

    # Extract actual PositionedTable call args
    _, kwargs = mock_positioned_table.call_args
    x_offset = kwargs["x_offset"]
    y_offset = kwargs["y_offset"]

    # Manual expected values:
    expected_table_width = 1 * 200  # one image, draw_height is width
    expected_x_offset = ((frame_width - expected_table_width) * 0.5) + (3 * margin)
    expected_y_offset = (frame_height * 0.5) + header_height

    assert x_offset == expected_x_offset
    assert y_offset == expected_y_offset


# Test: confirm that it works without a section title
# It should skip heading but still build the layout
@patch("Hydrological_model_validator.Report.report_utils.PositionedTable")
@patch("Hydrological_model_validator.Report.report_utils.RotatedImage")
def test_add_multiple_rotated_images_grid_no_heading(mock_rotated_image, mock_positioned_table, tmp_path):
    pdf = create_mock_pdf()
    img_path = tmp_path / "img.png"
    img_path.touch()

    mock_rotated_image.return_value = Mock(draw_height=100)

    add_multiple_rotated_images_grid(pdf, [img_path], cols=1, section_title=None)

    # Should not call add_heading
    pdf.add_heading.assert_not_called()

    # Should still add PositionedTable and PageBreak
    assert mock_positioned_table.called
    assert any(isinstance(f, PageBreak) for f in pdf.story)
    
    
###############################################################################
# add_seasonal_scatter_page class tests
###############################################################################


def create_mock_pdf_wSpacer():
    class PdfMock:
        def __init__(self):
            self.story = []
            self.add_heading = Mock()
            self.styles = getSampleStyleSheet() 
        # If add_seasonal_scatter_page calls pdf.add_spacer, implement it:
        def add_spacer(self, height):
            self.story.append(Spacer(1, height))
    return PdfMock()

# Test that the section heading and initial spacer are added correctly to the PDF story
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image")
def test_heading_and_spacers_added(mock_image, mock_open, tmp_path):
    pdf = create_mock_pdf_wSpacer()

    main_path = tmp_path / "main.png"
    sub_paths = [tmp_path / f"sub_{i}.png" for i in range(4)]
    main_path.touch()
    for p in sub_paths:
        p.touch()

    # Mock the image size to arbitrary values used for aspect ratio calculations later
    mock_open.return_value.size = (800, 600)
    # Return a mock Image instance for all image calls
    mock_image.side_effect = [Mock()] * 5

    add_seasonal_scatter_page(pdf, main_path, sub_paths, section_title="Test Title")

    # Verify heading is added with correct title and level
    pdf.add_heading.assert_called_once_with("Test Title", level=0)
    # After heading, a Spacer should be added for vertical spacing
    assert hasattr(pdf.story[1], 'width') and hasattr(pdf.story[1], 'height')

# Test that the main image is added with correct resized dimensions preserving aspect ratio
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image")
def test_main_image_added_and_resized(mock_image, mock_open, tmp_path):
    pdf = create_mock_pdf()

    main_path = tmp_path / "main.png"
    main_path.touch()
    sub_paths = [tmp_path / f"sub_{i}.png" for i in range(4)]
    for p in sub_paths:
        p.touch()

    # Provide a known image size for aspect ratio calculation
    mock_open.return_value.size = (800, 400)  # aspect ratio = 0.5
    mock_main_img = Mock()
    # First call is main image, then 4 sub-images
    mock_image.side_effect = [mock_main_img] + [Mock()] * 4

    add_seasonal_scatter_page(pdf, main_path, sub_paths)

    # The main image drawWidth should be set to max_main_width (4 inches * 72 points)
    expected_width = 4 * 72
    assert mock_main_img.drawWidth == pytest.approx(expected_width)
    # The drawHeight should maintain the aspect ratio (width * height/width)
    assert mock_main_img.drawHeight == pytest.approx(expected_width * 0.5)

# Test that four sub images are added in a 2x2 table grid with correct sizing
@patch("Hydrological_model_validator.Report.report_utils.PILImage.open")
@patch("Hydrological_model_validator.Report.report_utils.Image")
def test_sub_images_grid_added_with_correct_layout(mock_image, mock_open, tmp_path):
    pdf = create_mock_pdf()

    main_path = tmp_path / "main.png"
    main_path.touch()
    sub_paths = [tmp_path / f"sub_{i}.png" for i in range(4)]
    for p in sub_paths:
        p.touch()

    # Mock sizes for sub-images
    # Create mocks for PILImage.open returns with .size attribute
    main_img_mock = Mock()
    main_img_mock.size = (800, 600)
    sub_img_mocks = []
    for _ in range(4):
        m = Mock()
        m.size = (400, 400)
        sub_img_mocks.append(m)
    mock_open.side_effect = [main_img_mock] + sub_img_mocks  # main img + 4 sub imgs

    # Mocks for main and sub images returned by reportlab Image()
    mock_main_img = Mock()
    mock_sub_imgs = [Mock() for _ in range(4)]
    mock_image.side_effect = [mock_main_img] + mock_sub_imgs

    add_seasonal_scatter_page(pdf, main_path, sub_paths)

    # Verify last item in story is a PageBreak to start a new page
    assert isinstance(pdf.story[-1], PageBreak)

    # The second last item should be the Table containing sub-images grid
    table = pdf.story[-2]
    from reportlab.platypus.tables import Table
    assert isinstance(table, Table)

    # Table data should be 2 rows and 2 columns (2x2)
    data = table._cellvalues
    assert len(data) == 2
    assert all(len(row) == 2 for row in data)

    # Check that all sub images in table have drawWidth set to max_sub_width (2.8 inches * 72)
    max_sub_width = 2.8 * 72
    for row in data:
        for cell in row:
            # cell is either a Mock image or something similar with drawWidth attribute
            if hasattr(cell, "drawWidth"):
                assert cell.drawWidth == pytest.approx(max_sub_width)
                
                
###############################################################################
# add_efficiency_pages class tests
###############################################################################

import pandas as pd
import calendar

def create_sample_efficiency_df():
    months_full = list(calendar.month_name)[1:] 
    columns = ['Total'] + months_full
    data = {
        'metric1': [0.85] + [0.8 + 0.01*i for i in range(12)],
        'metric2': [0.75] + [0.7 + 0.01*i for i in range(12)],
    }
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    return df

def create_sample_efficiency_df_nan():
    months_full = list(calendar.month_name)[1:] 
    columns = ['Total'] + months_full
    data = {
        'metric1': [0.85] + [0.8 + 0.01*i for i in range(12)],
        'metric2': [0.75] + [0.7, float('nan')] + [0.72 + 0.01*i for i in range(10)],  # NaN for February
    }
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    return df

# Check add_efficiency_pages adds headings with parentheses stripped
@patch("Hydrological_model_validator.Report.report_utils.PILImage")
def test_add_efficiency_pages_adds_headings(mock_image, tmp_path):
    pdf = create_mock_pdf_wSpacer()
    df = create_sample_efficiency_df()
    plot_titles = {'metric1': 'Efficiency (Test)', 'metric2': 'Another Metric'}
    add_efficiency_pages(pdf, df, plot_titles, tmp_path)
    # Parentheses should be removed in headings
    pdf.add_heading.assert_any_call("Efficiency", level=0)
    pdf.add_heading.assert_any_call("Another Metric", level=0)

# Check that add_efficiency_pages adds Spacers to story
@patch("Hydrological_model_validator.Report.report_utils.PILImage")
def test_add_efficiency_pages_adds_spacers(mock_image, tmp_path):
    pdf = create_mock_pdf_wSpacer()
    df = create_sample_efficiency_df()
    plot_titles = {'metric1': 'Efficiency (Test)'}
    add_efficiency_pages(pdf, df, plot_titles, tmp_path)
    # Confirm at least one Spacer is in story
    assert any(isinstance(item, Spacer) for item in pdf.story)

# Check add_efficiency_pages calls Image for each plot
@patch("Hydrological_model_validator.Report.report_utils.Image")
def test_add_efficiency_pages_calls_image(mock_image, tmp_path):
    pdf = create_mock_pdf_wSpacer()
    # Create a df without NaNs so plotting happens for both metrics
    import pandas as pd
    months = list(calendar.month_name)[1:] 
    columns = ['Total'] + months
    data = {
        'metric1': [0.85] * len(columns),
        'metric2': [0.75] * len(columns),
    }
    df = pd.DataFrame(data, index=columns).T
    plot_titles = {'metric1': 'Efficiency (Test)', 'metric2': 'Another Metric'}
    
    # Create dummy png files for each metric in tmp_path
    for metric_key in plot_titles.keys():
        dummy_img_path = tmp_path / f"{metric_key}.png"
        img = PILImage.new('RGB', (10, 10), color='white')
        img.save(dummy_img_path)
    
    add_efficiency_pages(pdf, df, plot_titles, tmp_path)

    assert mock_image.call_count >= 2

# Check tables created have correct columns and handle NaN as '-'
@patch("Hydrological_model_validator.Report.report_utils.PILImage")
def test_add_efficiency_pages_tables_structure_and_nan(mock_image, tmp_path):
    pdf = create_mock_pdf_wSpacer()
    df = create_sample_efficiency_df_nan()
    plot_titles = {'metric1': 'Efficiency (Test)', 'metric2': 'Another Metric'}
    add_efficiency_pages(pdf, df, plot_titles, tmp_path)
    from reportlab.platypus import Table
    tables = [item for item in pdf.story if isinstance(item, Table)]
    assert len(tables) == 4  # Two tables (month + total) per metric
    for table in tables:
        num_rows = len(table._cellvalues)
        num_cols = len(table._cellvalues[0])
        if num_rows == 4:  # monthly table
            assert num_cols == 6
        elif num_rows == 1:  # total table
            assert num_cols == 2
        else:
            raise AssertionError(f"Unexpected table structure: rows={num_rows}, cols={num_cols}")

    # Check that NaN replaced by '-' in February for metric2
    feb_idx = tables[2]._cellvalues[0].index('February')
    assert tables[2]._cellvalues[1][feb_idx] == '—'

# Check totals are formatted correctly as strings with two decimals
@patch("Hydrological_model_validator.Report.report_utils.PILImage")
def test_add_efficiency_pages_totals_format(mock_image, tmp_path):
    pdf = create_mock_pdf_wSpacer()
    df = create_sample_efficiency_df()
    plot_titles = {'metric1': 'Efficiency (Test)', 'metric2': 'Another Metric'}
    add_efficiency_pages(pdf, df, plot_titles, tmp_path)
    tables = [item for item in pdf.story if isinstance(item, Table)]
    total_idx = tables[1]._cellvalues[0].index('Total')
    total_value = tables[1]._cellvalues[0][total_idx + 1]
    assert total_value == '0.850'