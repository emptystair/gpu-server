from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from enum import Enum


@dataclass
class TextRegion:
    """Text region with bounding box and confidence"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str
    confidence: float


@dataclass
class TextBlock:
    """Merged text regions forming a logical block"""
    regions: List[TextRegion]
    merged_text: str
    block_confidence: float
    reading_order: int


@dataclass
class Column:
    """Column structure for multi-column layouts"""
    x_start: int
    x_end: int
    blocks: List[TextBlock]


@dataclass
class PageMetadata:
    """Metadata for a page"""
    page_number: int
    width: int
    height: int
    dpi: int


@dataclass
class PageResult:
    """Formatted result for a single page"""
    page_number: int
    text: str
    words: List[Dict[str, Any]]  # Compatible with WordResult structure
    confidence: float
    processing_time_ms: float


@dataclass
class FormatConfig:
    """Configuration for result formatting"""
    merge_threshold: float = 0.8  # Proximity threshold for merging
    column_detection_enabled: bool = True
    min_confidence: float = 0.0
    preserve_whitespace: bool = True


@dataclass
class OCROutput:
    """Raw OCR output from PaddleOCR"""
    boxes: List[List[Tuple[int, int]]]  # Polygon points
    texts: List[str]
    confidences: List[float]


class ResultFormatter:
    """Format OCR results into structured output"""
    
    def __init__(self, format_config: FormatConfig):
        """Purpose: Initialize result formatting rules
        Dependencies: None (pure Python)
        Priority: CORE
        """
        self.config = format_config
        self.proximity_threshold = 20  # pixels
        self.line_height_threshold = 15  # pixels
        
    def format_page_results(
        self,
        ocr_outputs: List[OCROutput],
        page_metadata: PageMetadata
    ) -> PageResult:
        """Purpose: Format raw OCR output into structured result
        Calls:
            - _merge_text_blocks()
            - _apply_reading_order()
            - _calculate_page_confidence()
        Called by: ocr_service._format_results()
        Priority: CORE
        """
        # Convert OCR outputs to text regions
        text_regions = self._convert_to_regions(ocr_outputs)
        
        # Filter by confidence if configured
        if self.config.min_confidence > 0:
            text_regions = [r for r in text_regions if r.confidence >= self.config.min_confidence]
        
        # Merge adjacent regions into blocks
        text_blocks = self._merge_text_blocks(text_regions)
        
        # Apply reading order
        ordered_blocks = self._apply_reading_order(text_blocks)
        
        # Build final text and word list
        full_text = ""
        words = []
        
        for block in ordered_blocks:
            if full_text and self.config.preserve_whitespace:
                full_text += "\n"
            full_text += block.merged_text
            
            # Add words with bounding boxes
            for region in block.regions:
                words.append({
                    "text": region.text,
                    "bbox": region.bbox,
                    "confidence": region.confidence
                })
        
        # Calculate overall confidence
        confidences = [r.confidence for r in text_regions]
        page_confidence = self._calculate_page_confidence(confidences)
        
        return PageResult(
            page_number=page_metadata.page_number,
            text=full_text,
            words=words,
            confidence=page_confidence,
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def _convert_to_regions(self, ocr_outputs: List[OCROutput]) -> List[TextRegion]:
        """Convert OCR outputs to text regions"""
        regions = []
        
        for output in ocr_outputs:
            for box, text, conf in zip(output.boxes, output.texts, output.confidences):
                # Convert polygon to bounding box
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                regions.append(TextRegion(
                    bbox=bbox,
                    text=text,
                    confidence=conf
                ))
        
        return regions
    
    def _merge_text_blocks(
        self,
        text_regions: List[TextRegion]
    ) -> List[TextBlock]:
        """Purpose: Merge adjacent text regions into blocks
        Calls:
            - _calculate_proximity()
            - _check_alignment()
        Called by: format_page_results()
        Priority: CORE
        """
        if not text_regions:
            return []
        
        # Sort regions by position (top to bottom, left to right)
        sorted_regions = sorted(text_regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        
        blocks = []
        current_block_regions = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            current_region = sorted_regions[i]
            last_region = current_block_regions[-1]
            
            # Check if regions should be merged
            proximity = self._calculate_proximity(last_region, current_region)
            aligned = self._check_alignment(last_region, current_region)
            
            if proximity < self.proximity_threshold and aligned:
                current_block_regions.append(current_region)
            else:
                # Create block from current regions
                blocks.append(self._create_text_block(current_block_regions, len(blocks)))
                current_block_regions = [current_region]
        
        # Don't forget the last block
        if current_block_regions:
            blocks.append(self._create_text_block(current_block_regions, len(blocks)))
        
        return blocks
    
    def _create_text_block(self, regions: List[TextRegion], order: int) -> TextBlock:
        """Create a text block from regions"""
        merged_text = " ".join(r.text for r in regions)
        confidences = [r.confidence for r in regions]
        block_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return TextBlock(
            regions=regions,
            merged_text=merged_text,
            block_confidence=block_confidence,
            reading_order=order
        )
    
    def _apply_reading_order(
        self,
        text_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Purpose: Sort text blocks in reading order
        Calls:
            - _detect_columns()
            - _sort_by_position()
        Called by: format_page_results()
        Priority: CORE
        """
        if not self.config.column_detection_enabled:
            return self._sort_by_position(text_blocks)
        
        # Detect column layout
        columns = self._detect_columns(text_blocks)
        
        if len(columns) <= 1:
            # Single column, simple top-to-bottom ordering
            return self._sort_by_position(text_blocks)
        
        # Multi-column: order within columns, then by column
        ordered_blocks = []
        for column in sorted(columns, key=lambda c: c.x_start):
            column_blocks = self._sort_by_position(column.blocks)
            ordered_blocks.extend(column_blocks)
        
        # Update reading order
        for i, block in enumerate(ordered_blocks):
            block.reading_order = i
        
        return ordered_blocks
    
    def _calculate_page_confidence(
        self,
        word_confidences: List[float]
    ) -> float:
        """Purpose: Calculate overall page confidence score
        Calls: Statistical calculations
        Called by: format_page_results()
        Priority: CORE
        """
        if not word_confidences:
            return 0.0
        
        # Use weighted average giving more weight to lower confidence scores
        # This penalizes pages with some very low confidence regions
        sorted_conf = sorted(word_confidences)
        weights = [1.0 / (i + 1) for i in range(len(sorted_conf))]
        weighted_sum = sum(c * w for c, w in zip(sorted_conf, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _detect_columns(
        self,
        text_blocks: List[TextBlock]
    ) -> List[Column]:
        """Purpose: Detect multi-column layouts
        Calls:
            - Clustering algorithms
            - _validate_column_structure()
        Called by: _apply_reading_order()
        Priority: CORE
        """
        if len(text_blocks) < 2:
            return [Column(x_start=0, x_end=float('inf'), blocks=text_blocks)]
        
        # Get x-coordinates of all blocks
        x_positions = []
        for block in text_blocks:
            for region in block.regions:
                x_positions.extend([region.bbox[0], region.bbox[2]])
        
        if not x_positions:
            return [Column(x_start=0, x_end=float('inf'), blocks=text_blocks)]
        
        # Simple column detection based on gaps
        x_positions.sort()
        gaps = []
        
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap threshold
                gaps.append((x_positions[i-1], x_positions[i], gap))
        
        # If no significant gaps, single column
        if not gaps:
            return [Column(x_start=min(x_positions), x_end=max(x_positions), blocks=text_blocks)]
        
        # Find the largest gap as column separator
        gaps.sort(key=lambda g: g[2], reverse=True)
        
        # For now, support only 2-column detection
        if gaps:
            separator = (gaps[0][0] + gaps[0][1]) // 2
            
            left_blocks = []
            right_blocks = []
            
            for block in text_blocks:
                block_x = sum(r.bbox[0] for r in block.regions) / len(block.regions)
                if block_x < separator:
                    left_blocks.append(block)
                else:
                    right_blocks.append(block)
            
            columns = []
            if left_blocks:
                columns.append(Column(x_start=0, x_end=separator, blocks=left_blocks))
            if right_blocks:
                columns.append(Column(x_start=separator, x_end=float('inf'), blocks=right_blocks))
            
            return columns if self._validate_column_structure(columns) else [
                Column(x_start=0, x_end=float('inf'), blocks=text_blocks)
            ]
        
        return [Column(x_start=0, x_end=float('inf'), blocks=text_blocks)]
    
    def _validate_column_structure(self, columns: List[Column]) -> bool:
        """Validate detected columns are reasonable"""
        if len(columns) < 2:
            return False
        
        # Check each column has sufficient content
        for column in columns:
            if len(column.blocks) < 2:
                return False
        
        return True
    
    def format_for_export(
        self,
        page_results: List[PageResult],
        export_format: str
    ) -> Union[Dict, str]:
        """Purpose: Convert results to requested format
        Calls:
            - _to_json() if JSON
            - _to_plain_text() if TXT
            - _to_structured_data() if XML
        Called by: API response formatting
        Priority: CORE
        """
        if export_format.lower() == "json":
            return self._to_json(page_results)
        elif export_format.lower() == "txt":
            return self._to_plain_text(page_results)
        elif export_format.lower() == "xml":
            return self._to_structured_data(page_results)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _to_json(self, page_results: List[PageResult]) -> Dict:
        """Convert to JSON format"""
        return {
            "pages": [
                {
                    "page_number": pr.page_number,
                    "text": pr.text,
                    "confidence": pr.confidence,
                    "word_count": len(pr.words),
                    "processing_time_ms": pr.processing_time_ms
                }
                for pr in page_results
            ],
            "total_pages": len(page_results),
            "average_confidence": sum(pr.confidence for pr in page_results) / len(page_results) if page_results else 0.0
        }
    
    def _to_plain_text(self, page_results: List[PageResult]) -> str:
        """Convert to plain text format"""
        texts = []
        for pr in page_results:
            if pr.text.strip():
                texts.append(f"--- Page {pr.page_number} ---")
                texts.append(pr.text)
                texts.append("")
        
        return "\n".join(texts)
    
    def _to_structured_data(self, page_results: List[PageResult]) -> str:
        """Convert to XML format"""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<document>')
        
        for pr in page_results:
            xml_parts.append(f'  <page number="{pr.page_number}" confidence="{pr.confidence:.4f}">')
            xml_parts.append(f'    <text><![CDATA[{pr.text}]]></text>')
            xml_parts.append('  </page>')
        
        xml_parts.append('</document>')
        return '\n'.join(xml_parts)
    
    def _calculate_proximity(
        self,
        region1: TextRegion,
        region2: TextRegion
    ) -> float:
        """Purpose: Calculate distance between text regions
        Calls: Geometric calculations
        Called by: _merge_text_blocks()
        Priority: CORE
        """
        # Calculate minimum distance between bounding boxes
        # Horizontal distance
        h_dist = max(0, region2.bbox[0] - region1.bbox[2], region1.bbox[0] - region2.bbox[2])
        
        # Vertical distance
        v_dist = max(0, region2.bbox[1] - region1.bbox[3], region1.bbox[1] - region2.bbox[3])
        
        # Euclidean distance
        return (h_dist ** 2 + v_dist ** 2) ** 0.5
    
    def _check_alignment(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if two regions are aligned (same line or column)"""
        # Check horizontal alignment (same line)
        v_overlap = min(region1.bbox[3], region2.bbox[3]) - max(region1.bbox[1], region2.bbox[1])
        height1 = region1.bbox[3] - region1.bbox[1]
        height2 = region2.bbox[3] - region2.bbox[1]
        min_height = min(height1, height2)
        
        if v_overlap > 0.5 * min_height:
            return True
        
        # Check vertical alignment (same column)
        h_center1 = (region1.bbox[0] + region1.bbox[2]) / 2
        h_center2 = (region2.bbox[0] + region2.bbox[2]) / 2
        
        if abs(h_center1 - h_center2) < 50:  # Column alignment threshold
            return True
        
        return False
    
    def _sort_by_position(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort blocks by vertical position"""
        return sorted(blocks, key=lambda b: min(r.bbox[1] for r in b.regions))
    
    def apply_confidence_threshold(
        self,
        results: PageResult,
        threshold: float
    ) -> PageResult:
        """Purpose: Filter results by confidence
        Calls: Filter operations
        Called by: Optional post-processing
        Priority: CORE
        """
        filtered_words = [w for w in results.words if w["confidence"] >= threshold]
        
        # Rebuild text from filtered words
        filtered_text = " ".join(w["text"] for w in filtered_words)
        
        # Recalculate confidence
        new_confidence = sum(w["confidence"] for w in filtered_words) / len(filtered_words) if filtered_words else 0.0
        
        return PageResult(
            page_number=results.page_number,
            text=filtered_text,
            words=filtered_words,
            confidence=new_confidence,
            processing_time_ms=results.processing_time_ms
        )