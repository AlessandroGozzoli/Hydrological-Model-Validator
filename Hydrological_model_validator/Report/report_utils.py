###############################################################################
##                                                                           ##
##                               LIBRARIES                                   ##
##                                                                           ##
###############################################################################

# PDF document creation and structure
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Flowable,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

# PDF styling and layout
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage

# Image handling
from PIL import Image as PILImage

# Metadata and versioning
from importlib.metadata import version, PackageNotFoundError

# Date and time handling
from datetime import datetime

# Standard library utilities
import calendar
import re
from pathlib import Path

# Data handling
import pandas as pd

###############################################################################
##                                                                           ##
##                                CLASSES                                    ##
##                                                                           ##
###############################################################################


class MyDocTemplate(SimpleDocTemplate):
    """
    Custom document template that extends ReportLab's SimpleDocTemplate.

    This class overrides the afterFlowable method to detect headings in the
    flowables added to the document. When a heading with an outlineLevel attribute
    is encountered, it registers the heading with the Table of Contents (TOC)
    system by notifying the TOCEntry event. This enables automatic building
    of the Table of Contents with correct page numbers.

    Attributes:
        _headings (list): Internal list to keep track of headings added (optional use).
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize the custom document template.

        Args:
            filename (str): The path to the output PDF file.
            **kwargs: Additional keyword arguments passed to SimpleDocTemplate.
        """
        super().__init__(filename, **kwargs)
        # Initialize an internal list for headings if needed for tracking or future use
        self._headings = []

    def afterFlowable(self, flowable):
        """
        Hook method called by ReportLab after each flowable is processed.

        Checks if the flowable has an 'outlineLevel' attribute. If so,
        it considers the flowable as a heading and notifies the TOC system
        with the heading level, text content, and current page number.

        This notification allows the Table of Contents to be built dynamically
        during PDF generation.

        Args:
            flowable (Flowable): The flowable object just added to the document.
        """
        # Only process flowables that have an outlineLevel attribute, meaning they are headings
        if hasattr(flowable, 'outlineLevel') and flowable.outlineLevel is not None:
            level = flowable.outlineLevel  # Extract the heading level (e.g., 0 for main, 1 for subheading)
            text = flowable.getPlainText()  # Get plain text of the heading for TOC display
            page_num = self.page  # Current page number in the document for linking in TOC

            # Notify the ReportLab TOC system that a new heading was added
            # This allows the TOC to register the heading text, level, and page for automatic entries
            self.notify('TOCEntry', (level, text, page_num))

###############################################################################
            
###############################################################################

class PDFReportBuilder:
    """
    A builder class for creating PDF reports using ReportLab.

    This class manages the document structure, styles, and content 
    to generate professional-looking PDF reports with title pages, 
    table of contents, and formatted sections.

    Attributes:
        pdf_path (str): Path where the PDF will be saved.
        styles (StyleSheet1): ReportLab stylesheet object for styling text.
        story (list): List of flowables representing the PDF content.
        toc (TableOfContents): Table of contents flowable.
    """
    def __init__(self, pdf_path):
        """
        Initialize the PDFReportBuilder with the target output path.

        Sets up default and custom styles, configures table of contents styles,
        and prepares the empty story list for content.

        Args:
            pdf_path (str): Path for saving the generated PDF file.
        """
        self.pdf_path = pdf_path
        self.styles = getSampleStyleSheet()
        self.story = []

        # Center headings to improve title and section readability on the page
        self.styles['Heading1'].alignment = TA_CENTER
        self.styles['Heading2'].alignment = TA_CENTER

        # Add custom paragraph styles tailored for titles and section headers
        self.styles.add(ParagraphStyle(name='CenteredTitle', parent=self.styles['Title'], alignment=TA_CENTER))
        self.styles.add(ParagraphStyle(name='SectionTitle', fontSize=16, leading=20, spaceAfter=12))

        # Initialize a Table of Contents object with hierarchical styling for two heading levels
        # Styles define indentation and font to visually differentiate TOC levels
        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(fontName='Times-Bold', fontSize=14, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=5, leading=16),
            ParagraphStyle(fontSize=12, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=0, leading=12),
        ]

    def build_title_page(
            self,
            title="Model Evaluation Report",
            subtitle="Comprehensive model performance evaluation",
            author="Alessandro Gozzoli",
            organization="Alma Mater Studiorum - Università di Bologna",
            email="alessandro.gozzoli4@studio.unibo.it",
            cellphone="",
            logo_path=None,
            package_name="hydrological_model_validator"
            ):
        """
        Build and append the title page content to the story.

        This method resets the current story content and adds styled
        paragraphs including the title, subtitle, date, author info, version,
        and an optional logo image. Visual separators and spacing are also added
        for professional layout.

        Args:
            title (str): Main report title text.
            subtitle (str): Subtitle or descriptive text.
            author (str): Name of the report author.
            organization (str): Affiliated organization or institution.
            email (str): Contact email address.
            cellphone (str): Contact phone number (optional).
            logo_path (str or None): Path to an image file for logo inclusion (optional).
            package_name (str): Package name for version info lookup.

        Side Effects:
            Clears and appends multiple flowables (paragraphs, images, spacers, lines)
            to the internal story list representing the PDF content.
        """
        # Reset content to ensure title page is generated fresh without leftover elements
        self.story = []

        # Letter size dimensions used for centering and positioning elements on the page
        PAGE_WIDTH, PAGE_HEIGHT = letter

        # Define consistent colors to unify visual style and emphasize sections accordingly
        primary_color = HexColor("#2E5A99")    # Used for main titles to draw attention
        line_color = HexColor("#CCCCCC")       # Subtle lines for separation without distraction
        subtitle_color = HexColor("#555555")   # Medium gray for subtitles and meta info to avoid harsh contrast

        # Attempt to fetch package version to include in report footer, fallback if not installed
        try:
            pkg_version = version(package_name)
        except PackageNotFoundError:
            pkg_version = "Unknown"

        # Helper function for vertical spacing to keep layout adjustments straightforward
        def vspace(height):
            self.story.append(Spacer(1, height))

        # Define a Flowable subclass to draw horizontal lines consistently across the document
        # Lines visually separate sections, enhancing structure and readability
        class HorizontalLine(Flowable):
            def __init__(self, width, thickness=1, color=line_color):
                super().__init__()
                self.width = width
                self.thickness = thickness
                self.color = color
                self.height = thickness

            def draw(self):
                self.canv.setStrokeColor(self.color)
                self.canv.setLineWidth(self.thickness)
                self.canv.line(0, self.height / 2, self.width, self.height / 2)

        # Conditionally add a logo image, scaled and centered to enhance brand identity without disrupting layout
        if logo_path:
            try:
                img = Image(str(logo_path))
                img.drawHeight = 1.0 * inch
                img.drawWidth = 1.0 * inch
                self.story.append(Spacer(1, 0.5 * inch))  # Add space above logo for balance
                self.story.append(img)
                vspace(0.2 * inch)  # Small gap after logo for visual separation
            except Exception as e:
                # Failure to load logo does not prevent report generation; warn user instead
                print(f"Warning: Could not load logo at {logo_path}: {e}")

        # Insert a horizontal line near the top to anchor the title visually
        self.story.append(HorizontalLine(PAGE_WIDTH - 2*inch, thickness=1.5))
        vspace(0.3 * inch)

        # Define a bold, large font style for the main title to establish hierarchy and attract attention
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=self.styles['Title'],
            alignment=TA_CENTER,
            fontSize=36,
            leading=42,
            textColor=primary_color,
            spaceAfter=12,
            spaceBefore=12,
            fontName="Helvetica-Bold"
        )
        self.story.append(Paragraph(title, title_style))

        # Subtitle in italic and smaller font provides context without overshadowing title
        subtitle_style = ParagraphStyle(
            name="SubtitleStyle",
            parent=self.styles['Normal'],
            alignment=TA_CENTER,
            fontSize=16,
            leading=20,
            textColor=subtitle_color,
            italic=True,
            spaceAfter=6
        )
        self.story.append(Paragraph(subtitle, subtitle_style))

        # Display current date to timestamp report generation
        date_str = datetime.now().strftime("%B %d, %Y")
        date_style = ParagraphStyle(
            name="DateStyle",
            parent=self.styles['Normal'],
            alignment=TA_CENTER,
            fontSize=12,
            leading=14,
            textColor=subtitle_color,
            spaceAfter=6
        )
        self.story.append(Paragraph(f"Date: {date_str}", date_style))

        # Version info shows the tool's version generating the report, providing traceability
        version_style = ParagraphStyle(
            name="VersionStyle",
            parent=self.styles['Normal'],
            alignment=TA_CENTER,
            fontSize=12,
            leading=14,
            textColor=subtitle_color,
            spaceAfter=18
        )
        self.story.append(Paragraph(f"Report generated by Hydrological Model Validator v{pkg_version}", version_style))
        
        # Another horizontal line to visually separate the header section from the footer info block
        vspace(0.4 * inch)
        self.story.append(HorizontalLine(PAGE_WIDTH - 2*inch, thickness=1.5))

        # Push author/contact information to the bottom of the page by adding flexible vertical space
        self.story.append(Spacer(1, PAGE_HEIGHT - 6*inch))

        # Footer style for contact info is smaller and subtle to keep focus on main content
        info_style = ParagraphStyle(
            name="InfoStyle",
            parent=self.styles['Normal'],
            alignment=TA_CENTER,
            fontSize=10,
            leading=12,
            textColor=subtitle_color,
            spaceAfter=4
        )

        # Add author, organization, and contact details to provide proper attribution and contact info
        self.story.append(Paragraph(f"<b>Author:</b> {author}", info_style))
        self.story.append(Paragraph(f"<b>Organization:</b> {organization}", info_style))
        self.story.append(Spacer(1, 12))  # Adds visual gap before contacts
        self.story.append(Paragraph("<b>Contacts:</b>", info_style))
        self.story.append(Paragraph(email, info_style))
        self.story.append(Paragraph(cellphone, info_style))
        
        # Finalize the title page by forcing a page break, isolating it from following content
        self.story.append(PageBreak())
        
    def build_toc(self):
        """
        Build and append a styled Table of Contents page to the story.

        This method adds a title with custom styling, a horizontal line,
        and the TableOfContents flowable configured in the constructor.
        A page break is appended at the end to separate the TOC from following content.

        Side Effects:
            Appends flowables to the internal story list representing the PDF content.
        """

        PAGE_WIDTH, _ = letter
        primary_color = HexColor("#2E5A99")
        line_color = HexColor("#CCCCCC")
    
        # Create a visually distinct title for the Table of Contents page for clarity and structure
        toc_title_style = ParagraphStyle(
            name="TOCTitleStyle",
            parent=self.styles['Heading1'],
            alignment=TA_CENTER,
            fontSize=28,
            leading=32,
            textColor=primary_color,
            spaceAfter=12,
            spaceBefore=24,
            fontName="Helvetica-Bold"
        )
        self.story.append(Paragraph("Table of Contents", toc_title_style))

        # Horizontal line below TOC title visually separates header from content
        class HorizontalLine(Flowable):
            def __init__(self, width, thickness=1.5, color=line_color):
                super().__init__()
                self.width = width
                self.thickness = thickness
                self.color = color
                self.height = thickness

            def draw(self):
                self.canv.setStrokeColor(self.color)
                self.canv.setLineWidth(self.thickness)
                self.canv.line(0, self.height / 2, self.width, self.height / 2)

        self.story.append(HorizontalLine(PAGE_WIDTH - 2 * inch))
        self.story.append(Spacer(1, 20))
    
        # Insert the TOC flowable object which dynamically generates entries from added headings
        self.story.append(self.toc)
        
        # End TOC page with a break to start new content cleanly on the following page
        self.story.append(PageBreak())
            
    def add_heading(self, text, level=0):
        """
        Add a heading paragraph to the story with the specified level.

        This method creates a paragraph with either Heading1 or Heading2 style,
        assigns a unique bookmark name, and sets outlineLevel and bookmark attributes
        to enable navigation and inclusion in the Table of Contents.

        Args:
            text (str): The heading text.
            level (int): Heading level (0 for main heading, 1 for subheading).
        """
        # Select heading style based on level; level 0 is major, level 1 is subheading
        style_name = 'Heading1' if level == 0 else 'Heading2'
        style = self.styles[style_name]
        
        # Unique bookmark name ensures that each heading can be individually referenced in the document outline and TOC
        bookmark_name = f"heading_{len(self.story)}"
        
        # Create a paragraph for the heading, attaching metadata for outline level and bookmark to integrate with TOC
        para = Paragraph(text, style)
        setattr(para, "outlineLevel", level)
        setattr(para, "bookmark", bookmark_name)

        # Add the heading paragraph to the story to be rendered in the PDF
        self.story.append(para)

    def save(self):
        doc = MyDocTemplate(self.pdf_path, pagesize=letter)
        doc.multiBuild(self.story)

###############################################################################

###############################################################################
        
class PositionedTable(Flowable):
    """
    A wrapper Flowable that allows precise positioning of another Flowable
    (e.g., a Table) by applying horizontal and vertical offsets when drawn.

    This is useful when you need to control the exact position of a table
    or any flowable element on the canvas, beyond the normal flow layout.

    Attributes:
        flowable (Flowable): The flowable object to be positioned.
        x_offset (float): Horizontal shift from the current origin in points.
        y_offset (float): Vertical shift from the current origin in points.
    """

    def __init__(self, flowable, x_offset=0, y_offset=0):
        """
        Initialize the PositionedTable with the flowable to position and offsets.

        Args:
            flowable (Flowable): The flowable element to wrap and position.
            x_offset (float, optional): Horizontal offset in points (default 0).
            y_offset (float, optional): Vertical offset in points (default 0).
        """
        super().__init__()
        self.flowable = flowable
        self.x_offset = x_offset
        self.y_offset = y_offset

    def wrap(self, availWidth, availHeight):
        """
        Calculate the space required by the wrapped flowable.

        Delegates the wrap calculation to the wrapped flowable, since the
        positioning offsets do not affect the required size.

        Args:
            availWidth (float): Available width for the flowable.
            availHeight (float): Available height for the flowable.

        Returns:
            (width, height): The size needed by the wrapped flowable.
        """
        # Use the wrapped flowable's own wrap method to get its required size
        return self.flowable.wrap(availWidth, availHeight)

    def draw(self):
        """
        Draw the wrapped flowable at the specified offsets.

        The canvas state is saved and restored to avoid affecting other drawings.
        The canvas origin is translated by the given offsets, then the flowable
        is drawn at the translated origin (0, 0), effectively positioning it
        precisely relative to the original drawing point.
        """
        self.canv.saveState()  # Save the current canvas state (position, styles)
        # Move the origin by x_offset horizontally and y_offset vertically
        self.canv.translate(self.x_offset, self.y_offset)
        # Draw the wrapped flowable at the new origin (0,0) after translation
        self.flowable.drawOn(self.canv, 0, 0)
        self.canv.restoreState()  # Restore the canvas state to avoid side effects

###############################################################################

###############################################################################
        
class RotatedImage(Flowable):
    """
    A Flowable that loads an image from disk and draws it rotated by 90 degrees
    within a fixed frame size, scaling the image to fit while preserving aspect ratio.

    The image is rotated clockwise by 90 degrees and positioned inside a rectangular
    frame defined by frame_width and frame_height, with margins and optional vertical offset.

    Attributes:
        img_path (str): Path to the image file.
        frame_width (float): Width of the frame where the image is drawn (in points).
        frame_height (float): Height of the frame where the image is drawn (in points).
        margin (float): Margin around the image inside the frame (in points).
        header_height (float): Additional space reserved for a header above the image (in points).
        custom_offset_y (float or None): Optional custom vertical offset for positioning.
        orig_width (int): Original width of the image in pixels.
        orig_height (int): Original height of the image in pixels.
        draw_width (float): Width of the scaled image for drawing (in points).
        draw_height (float): Height of the scaled image for drawing (in points).
    """

    def __init__(self, img_path, frame_width=456.0, frame_height=636.0, 
                 margin=0.1 * inch, header_height=0.25 * inch, custom_offset_y=None):
        """
        Initialize the RotatedImage with path, frame size, margins, and optional vertical offset.

        Args:
            img_path (str): Path to the image file to load and draw.
            frame_width (float, optional): Width of the frame box (default 456 points).
            frame_height (float, optional): Height of the frame box (default 636 points).
            margin (float, optional): Margin around the image inside the frame (default 0.1 inch).
            header_height (float, optional): Reserved space for header above the image (default 0.25 inch).
            custom_offset_y (float or None, optional): Override vertical positioning if provided.
        """
        super().__init__()
        self.img_path = img_path
        self.margin = margin
        self.header_height = header_height
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.custom_offset_y = custom_offset_y

        # Open the image to get its original size (width, height) in pixels
        with PILImage.open(img_path) as img:
            self.orig_width, self.orig_height = img.size

        # Calculate the max dimensions available for the rotated image inside the frame
        # Note: The image will be rotated 90°, so width and height swap roles for scaling
        max_rotated_width = frame_width - 2 * margin
        max_rotated_height = frame_height - 2 * margin - header_height

        # Calculate scale factors based on the rotated image's dimensions:
        # Since the image is rotated, the original height corresponds to the width after rotation, and vice versa
        scale_w = max_rotated_width / self.orig_height  # width available / image height
        scale_h = max_rotated_height / self.orig_width  # height available / image width

        # Choose the smaller scale to fit the image entirely within the frame without distortion
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale if image is smaller than frame

        # Calculate the final drawn dimensions after scaling, keeping original orientation (unrotated)
        self.draw_height = self.orig_height * scale  # height before rotation
        self.draw_width = self.orig_width * scale    # width before rotation

    def wrap(self, availWidth, availHeight):
        """
        Calculate the space required by the rotated image with margins and header.

        The required width is the scaled image width plus horizontal margins.
        The required height is the scaled image height plus vertical margins and header space.

        Args:
            availWidth (float): Available width for layout (ignored here).
            availHeight (float): Available height for layout (ignored here).

        Returns:
            tuple: (required width, required height) for layout.
        """
        wrapped_width = self.draw_width + 2 * self.margin
        wrapped_height = self.draw_height + 2 * self.margin + self.header_height
        return wrapped_width, wrapped_height

    def draw(self):
        """
        Draw the rotated image on the canvas.

        The image is rotated by 90 degrees clockwise and translated so that it fits
        centered within the frame, considering margins and optional vertical offset.

        The canvas state is saved and restored to prevent side effects on subsequent drawings.
        """

        self.canv.saveState()  # Save the current canvas state before transformations

        # Calculate horizontal offset to center the rotated image inside the frame width
        # After rotation, image width is 'draw_height' because of 90° rotation
        offset_x = (self.frame_width - self.draw_height) / 2

        # Determine vertical offset:
        # Use the provided custom offset if available;
        # otherwise, compute a default offset that accounts for header height and image width after rotation
        if self.custom_offset_y is not None:
            offset_y = self.custom_offset_y
        else:
            # Complex offset calculation likely tuned for specific layout needs
            offset_y = (self.frame_height + (self.header_height * 7) - self.draw_width) * 2

        # Move the origin to the calculated offset position before rotation
        self.canv.translate(offset_x, -offset_y)

        # Rotate the canvas coordinate system by 90 degrees clockwise
        self.canv.rotate(90)

        # After rotation, translate upwards by the image height to align drawing origin
        self.canv.translate(0, -self.draw_height)

        # Create a ReportLab Image flowable for the image file
        img = RLImage(self.img_path)
        # Set the image size to the scaled dimensions
        img.drawHeight = self.draw_height
        img.drawWidth = self.draw_width
        img.wrapOn(self.canv, self.draw_width, self.draw_height)
        # Draw the image at the transformed origin (0, 0)
        img.drawOn(self.canv, 0, 0)

        self.canv.restoreState()  # Restore canvas to original state after drawing

      
###############################################################################
##                                                                           ##
##                               FUNCTIONS                                   ##
##                                                                           ##
###############################################################################


def add_plot_to_pdf(pdf, img_path, section_title, width=6*inch):
    """
    Add a titled image section to a PDF document with proper spacing and page break.

    This function inserts a section heading followed by an image into a PDF report. It
    adjusts the image size to a specified width while maintaining its aspect ratio.
    Spacers are added before and after the image to improve layout, and a page break
    is appended after the image to separate sections cleanly.

    Parameters
    ----------
    pdf : ReportLab PDF document object
        The PDF document to which the image and heading are added. Must support
        methods like add_heading and have a story attribute (a list of flowables).
    img_path : str or pathlib.Path
        Path to the image file to insert into the PDF.
    section_title : str
        Title text to be added as a heading before the image.
    width : float, optional
        Desired width of the image in points (default is 6 inches).

    Returns
    -------
    None
        The function modifies the PDF object in-place by adding flowables to its story.

    Example
    -------
    >>> add_plot_to_pdf(pdf, "plots/plot1.png", "Section 1: Overview")
    """

    # Add section heading at level 0 (top level heading)
    pdf.add_heading(section_title, level=0)  
    
    flowables = []

    # Add vertical space before the image
    flowables.append(Spacer(1, 0.2 * inch))

    # Load image using ReportLab's Image flowable
    img = Image(str(img_path))

    # Get original image dimensions (width, height) from PIL
    orig_width, orig_height = PILImage.open(img_path).size

    # Calculate aspect ratio to preserve image proportions
    aspect_ratio = orig_height / orig_width

    # Set image width to desired width, adjust height to maintain aspect ratio
    img.drawWidth = width
    img.drawHeight = width * aspect_ratio

    # Append the image to flowables
    flowables.append(img)

    # Add more vertical space after the image
    flowables.append(Spacer(1, 0.5 * inch))

    # Extend the PDF story with the flowables (heading, spacer, image, spacer)
    pdf.story.extend(flowables)

    # Add a page break after the image section to separate from next content
    pdf.story.append(PageBreak())

###############################################################################

###############################################################################

def add_rotated_image_page(pdf, img_path, section_title=None, custom_offset_y=None):
    """
    Add a page with a rotated image to the PDF document, optionally with a section heading.

    This function inserts an optional section heading followed by a rotated image that fits
    a predefined frame size. The image is rotated 90 degrees clockwise and positioned with
    optional vertical offset adjustment. A page break is added after the image to separate
    it from subsequent content.

    Parameters
    ----------
    pdf : ReportLab PDF document object
        The PDF document to which the rotated image and optional heading are added. Must
        support add_heading and have a story attribute (list of flowables).
    img_path : str or pathlib.Path
        Path to the image file to be added, which will be rotated and scaled to fit.
    section_title : str, optional
        Title text to add as a heading before the image. If None, no heading is added.
    custom_offset_y : float, optional
        Custom vertical offset for positioning the rotated image. Overrides default positioning.

    Returns
    -------
    None
        Modifies the PDF object in-place by appending flowables to its story.

    Example
    -------
    >>> add_rotated_image_page(pdf, "figures/diagram.png", "Rotated Diagram")
    """
    from reportlab.platypus import Spacer, PageBreak
    from reportlab.lib.units import inch

    # Add heading if provided, with some vertical spacing after
    if section_title:
        pdf.add_heading(section_title, level=0)
        pdf.story.append(Spacer(1, 0.3 * inch))

    # Create a RotatedImage flowable instance with optional vertical offset
    rotated_img = RotatedImage(str(img_path), custom_offset_y=custom_offset_y)

    # Append the rotated image to the PDF story
    pdf.story.append(rotated_img)

    # Add a page break to separate this page from the next content
    pdf.story.append(PageBreak())

###############################################################################

###############################################################################

def add_tables_page(
    pdf,
    tables_dict,
    section_title="Summary Tables",
    columns=2,
    rows=None,
    cell_padding=6,
    spacing=0.2 * inch,
    max_table_width=3.0 * inch,
):
    """
    Add a page with multiple tables arranged in a grid layout to the PDF document.

    This function creates a grid of summary tables, each with a title and metric-value pairs.
    The tables are arranged in a specified number of columns per row, with styling applied
    for readability. If a section title is provided, it is added at the top. The function
    handles incomplete rows by padding with empty spaces. A horizontal separator line is
    added after the first row if multiple rows exist. Finally, a page break is appended.

    Parameters
    ----------
    pdf : PDF document object
        PDF builder object which must have `.story` (a list of flowables) and `.add_heading` method.
    tables_dict : dict of str to dict of str to float
        Dictionary mapping each table title (str) to a dictionary of metric names and values.
    section_title : str, optional
        Title for the entire page section containing the tables. Default is "Summary Tables".
    columns : int, optional
        Number of tables to display per row. Default is 2.
    rows : int or None, optional
        Optional maximum number of rows to display. If None, all tables are shown.
    cell_padding : int, optional
        Padding inside each table cell in points. Default is 6.
    spacing : float, optional
        Vertical spacing between elements, in points. Default is 0.2 inch.
    max_table_width : float, optional
        Maximum width for each individual table in points. Default is 3.0 inch.

    Returns
    -------
    None
        The function modifies the `pdf` object in place by appending flowables.

    Example
    -------
    >>> tables = {
    ...     "Correlation Metrics": {"NSE": 0.85, "RMSE": 1.23},
    ...     "Efficiency Metrics": {"KGE": 0.90, "R2": 0.88},
    ... }
    >>> add_tables_page(pdf, tables, columns=2)
    """
    from reportlab.platypus import (
        Table as RLTable, TableStyle, Spacer, Paragraph
    )

    # Get base styles for text
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    # Define header style for table column headers (bold, centered)
    header_style = ParagraphStyle(
        name="TableHeader",
        parent=styles["Normal"],
        alignment=1,  # center alignment
        fontSize=10,
        spaceAfter=2,
        leading=12,
        fontName="Helvetica-Bold"
    )

    # Define style for table titles (larger, bold, centered)
    title_style = ParagraphStyle(
        name="TableTitle",
        parent=styles["Normal"],
        alignment=1,  # center
        fontSize=11,
        leading=13,
        fontName="Helvetica-Bold"
    )

    # Add section title heading and spacing if provided
    if section_title:
        pdf.add_heading(section_title, level=0)
        pdf.story.append(Spacer(1, spacing))

    # Build flowables for each table: title, spacer, table, spacer
    table_blocks = []
    for table_title, table_data in tables_dict.items():
        # Create a paragraph for the table title
        title_para = Paragraph(f"<b>{table_title}</b>", title_style)

        # Prepare the table data with column headers
        data = [
            [Paragraph("Metric", header_style), Paragraph("Correlation", header_style)]
        ]

        # Append each metric-value row with normal style
        for key, val in table_data.items():
            data.append([
                Paragraph(str(key), normal_style),
                Paragraph(f"{val:.4f}", normal_style)
            ])

        # Create ReportLab table with fixed column widths proportional to max_table_width
        tbl = RLTable(data, colWidths=[max_table_width * 0.6, max_table_width * 0.4])

        # Apply table styling: grid, background color for header, alignment, and padding
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightsteelblue),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), cell_padding),
            ("RIGHTPADDING", (0, 0), (-1, -1), cell_padding),
            ("TOPPADDING", (0, 0), (-1, -1), cell_padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), cell_padding),
        ]))

        # Append the title, small spacer, table, and larger spacer as a block
        table_blocks.append([title_para, Spacer(1, 0.1 * inch), tbl, Spacer(1, 0.25 * inch)])

    # Split tables into rows of 'columns' number of tables
    chunks = [table_blocks[i:i + columns] for i in range(0, len(table_blocks), columns)]

    for row_idx, row in enumerate(chunks):
        # Pad the last row with empty spacers if it has fewer than 'columns' tables
        while len(row) < columns:
            row.append([Spacer(1, max_table_width)] * 4)  # fill all 4 layers with spacer

        # Transpose the blocks so we can add all titles in one table row, then all spacers, etc.
        for layer in zip(*row):  # Each layer corresponds to title, spacer, table, spacer
            pdf.story.append(RLTable([list(layer)], colWidths=[max_table_width] * columns, hAlign="CENTER"))
            pdf.story.append(Spacer(1, 0.05 * inch))

        # Add horizontal separator line after the first row if multiple rows exist
        if row_idx == 0 and len(chunks) > 1:
            pdf.story.append(Spacer(1, 0.2 * inch))
            pdf.story.append(HRFlowable(width="100%", color=colors.grey, thickness=1))
            pdf.story.append(Spacer(1, 0.3 * inch))

    # Add a page break at the end to separate from next page content
    pdf.story.append(PageBreak())

###############################################################################

###############################################################################

def add_multiple_images_grid(pdf, img_paths, section_title, columns=2, rows=None, max_width=3*inch, spacing=0.2*inch):
    """
    Add multiple images to the PDF in a neat grid layout with optional limits on rows and columns.

    Parameters
    ----------
    pdf : PDF builder object
        The PDF document builder that supports `.story` (list of flowables) and `.add_heading`.
    img_paths : list of str or pathlib.Path
        List of file paths to images to be added.
    section_title : str
        Title text displayed above the image grid.
    columns : int, optional
        Number of columns in the grid (default is 2).
    rows : int or None, optional
        Maximum number of rows to include; if None, include all images.
    max_width : float, optional
        Maximum width of each image in points (default 3 inches).
    spacing : float, optional
        Vertical spacing above and below the image grid (default 0.2 inch).

    Returns
    -------
    None
        Modifies the `pdf` object in place by appending image grid and spacing flowables.
    """

    # Add the section heading and top spacing
    pdf.add_heading(section_title, level=0)
    pdf.story.append(Spacer(1, spacing))

    # Load images, resize while maintaining aspect ratio, and collect as flowables
    images = []
    for img_path in img_paths:
        img = Image(str(img_path))
        orig_width, orig_height = PILImage.open(img_path).size
        aspect_ratio = orig_height / orig_width
        img.drawWidth = max_width
        img.drawHeight = max_width * aspect_ratio
        images.append(img)

    # If row limit specified, truncate images accordingly
    if rows is not None:
        images = images[:rows * columns]

    # Pad images list so last row is complete with invisible Spacers
    while len(images) % columns != 0:
        images.append(Spacer(1, 0.1 * inch))

    # Break images into rows for the table layout
    table_data = [images[i:i + columns] for i in range(0, len(images), columns)]

    # Create a ReportLab Table to hold the image grid
    table = Table(table_data, hAlign='CENTER')

    # Apply padding and center alignment to the table cells
    table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    # Append the image grid table, bottom spacing, and a page break
    pdf.story.append(table)
    pdf.story.append(Spacer(1, spacing))
    pdf.story.append(PageBreak())

###############################################################################

###############################################################################

def add_multiple_rotated_images_grid(
        pdf,
        img_paths,
        cols=2,
        section_title=None,
        margin=0.25 * inch,
        frame_width=456.0,
        frame_height=636.0,
        header_height=0.5 * inch,
    ):
    """
    Adds a grid of rotated images to the PDF, arranging them in specified columns,
    centered within a defined frame size, with optional section title.

    Parameters
    ----------
    pdf : PDF builder object
        The PDF document builder supporting `.story` and `.add_heading`.
    img_paths : list of str or pathlib.Path
        Paths to images that will be rotated and added.
    cols : int, optional
        Number of columns in the grid (default 2).
    section_title : str or None, optional
        Optional heading displayed above the image grid.
    margin : float, optional
        Margin (in points) around each rotated image (default 0.25 inch).
    frame_width : float, optional
        Width of the frame container for the grid (default 456 points).
    frame_height : float, optional
        Height of the frame container for the grid (default 636 points).
    header_height : float, optional
        Space allocated for header above the grid (default 0.5 inch).

    Returns
    -------
    None
        Appends positioned rotated images grid and page break to `pdf.story`.
    """
    

    if section_title:
        pdf.add_heading(section_title, level=0)
        pdf.story.append(Spacer(1, 0.2 * inch))

    # Create rotated image flowables with specified frame and margin
    rotated_images = [
        RotatedImage(
            str(path),
            frame_width=frame_width,
            frame_height=frame_height,
            margin=margin,
            header_height=header_height
        )
        for path in img_paths
    ]

    # Because images are rotated, width is draw_height and height is draw_width
    cell_widths = [img.draw_height for img in rotated_images[:cols]]  # widths of first row

    max_cell_width = max(cell_widths) if cell_widths else 0

    # Total table dimensions
    table_width = cols * max_cell_width

    # Arrange images into rows
    table_data = [rotated_images[i:i + cols] for i in range(0, len(rotated_images), cols)]

    # Pad last row with invisible spacers if incomplete
    if len(table_data[-1]) < cols:
        table_data[-1].extend([Spacer(1, 1)] * (cols - len(table_data[-1])))

    # Create ReportLab table for layout
    table = Table(table_data, hAlign='LEFT', spaceBefore=0, spaceAfter=0)

    # Style: center alignment, no padding for tight layout
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    # Calculate offsets to center the table within the frame + header space
    x_offset = ((frame_width - table_width) * 0.5) + (3 * margin)
    y_offset = ((frame_height * 0.5) + header_height)

    # Wrap and position the table flowable with offsets
    positioned = PositionedTable(table, x_offset=x_offset, y_offset=y_offset)

    pdf.story.append(positioned)
    pdf.story.append(PageBreak())

###############################################################################

###############################################################################

def add_seasonal_scatter_page(pdf, main_img_path, sub_img_paths, section_title="Seasonal Scatterplots"):
    """
    Adds a page to the PDF with a main seasonal scatterplot image on top,
    followed by a 2x2 grid of smaller subplots below.

    Args:
        pdf: PDF builder object with `.story` and `.add_heading`.
        main_img_path: Path to the main seasonal plot image.
        sub_img_paths: List of 4 image paths for the 2x2 grid of subplots.
        section_title: Title of the page section.
    """
    getSampleStyleSheet()
    pdf.add_heading(section_title, level=0)
    pdf.story.append(Spacer(1, 0.2 * inch))

    # --- 1. Add main seasonal plot ---
    max_main_width = 4 * inch

    main_img = Image(str(main_img_path))
    orig_width, orig_height = PILImage.open(main_img_path).size
    aspect_ratio = orig_height / orig_width
    main_img.drawWidth = max_main_width
    main_img.drawHeight = max_main_width * aspect_ratio

    pdf.story.append(main_img)
    pdf.story.append(Spacer(1, 0.1 * inch))

    # --- 2. Add 2x2 grid of subplots ---
    max_sub_width = 2.8 * inch

    sub_images = []
    for img_path in sub_img_paths:
        img = Image(str(img_path))
        orig_width, orig_height = PILImage.open(img_path).size
        aspect_ratio = orig_height / orig_width
        img.drawWidth = max_sub_width
        img.drawHeight = max_sub_width * aspect_ratio
        sub_images.append(img)

    # Build 2x2 table (assuming 4 subplots)
    table_data = [sub_images[i:i + 2] for i in range(0, len(sub_images), 2)]
    table = Table(table_data, hAlign='CENTER', spaceBefore=1)

    pdf.story.append(table)
    pdf.story.append(PageBreak())
    
###############################################################################

###############################################################################

def add_efficiency_pages(pdf, efficiency_df, plot_titles, plots_path):
    """
    Adds multiple pages to the PDF, each with a section for an efficiency metric:
    a heading, the corresponding plot image, a monthly values table, and a total value table.

    Args:
        pdf: PDF builder object with `.story`, `.add_heading`, and `.styles`.
        efficiency_df: pandas DataFrame indexed by metric keys, columns ['Total', month names].
        plot_titles: dict mapping metric_key to human-readable title.
        plots_path: Path to directory containing plot images named '<metric_key>.png'.
    """
    months = list(calendar.month_name)[1:]  # January to December

    for metric_key, title in plot_titles.items():
        # Clean title (remove any parenthesis and content inside)
        clean_title = re.sub(r'\s*\([^)]*\)', '', title)
        pdf.add_heading(clean_title, level=0)
        pdf.story.append(Spacer(1, 0.2 * inch))

        # --- Add Plot Image ---
        plot_path = Path(plots_path) / f"{metric_key}.png"
        if plot_path.exists():
            img = Image(str(plot_path), width=6 * inch, height=3.5 * inch)
            pdf.story.append(img)
        else:
            pdf.story.append(Paragraph(f"[Missing plot for {metric_key}]", pdf.styles['Normal']))
        pdf.story.append(Spacer(1, 0.3 * inch))

        # --- Prepare Monthly Table Data ---
        values = efficiency_df.loc[metric_key, ['Total'] + months].values
        total_val = values[0]
        month_vals = values[1:]

        # Two rows of 6 months each for months and their values
        month_table_data = [
            months[:6],
            [f"{v:.3f}" if pd.notna(v) else "—" for v in month_vals[:6]],
            months[6:],
            [f"{v:.3f}" if pd.notna(v) else "—" for v in month_vals[6:]],
        ]

        month_table = Table(month_table_data, hAlign='CENTER')
        month_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightsteelblue),   # First month label row
            ('BACKGROUND', (0, 2), (-1, 2), colors.lightsteelblue),   # Second month label row
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        pdf.story.append(month_table)
        pdf.story.append(Spacer(1, 0.2 * inch))

        # --- Prepare Total Table ---
        total_table = Table(
            [['Total', f"{total_val:.3f}" if pd.notna(total_val) else "—"]],
            colWidths=[2 * inch, 1.5 * inch], hAlign='CENTER'
        )
        total_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        pdf.story.append(total_table)
        pdf.story.append(PageBreak())