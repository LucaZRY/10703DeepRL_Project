#!/usr/bin/env python3
"""
PDF Guide Generator for Expert Data Preparation

This script generates a comprehensive PDF guide from the markdown documentation.
Requires: pip install reportlab markdown beautifulsoup4

Usage: python generate_pdf_guide.py
"""

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.platypus import Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import markdown
    from bs4 import BeautifulSoup
    import re
    from datetime import datetime
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install reportlab markdown beautifulsoup4")
    exit(1)


def create_pdf_guide():
    """Create a comprehensive PDF guide for expert data preparation."""

    # Create PDF document
    filename = "Expert_Data_Preparation_Guide.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)

    # Get styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue,
        borderWidth=1,
        borderColor=colors.darkblue,
        borderPadding=5,
        backColor=colors.lightblue
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkgreen
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        backgroundColor=colors.lightgrey,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=5,
        spaceAfter=10
    )

    # Build document content
    content = []

    # Title page
    content.append(Paragraph("Expert Data Preparation Guide", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Deep Reinforcement Learning - CarRacing Environment",
                             ParagraphStyle('Subtitle', parent=styles['Normal'],
                                          fontSize=16, alignment=TA_CENTER,
                                          textColor=colors.darkblue)))
    content.append(Spacer(1, 30))
    content.append(Paragraph("10703 Deep Reinforcement Learning Project",
                             ParagraphStyle('Subtitle2', parent=styles['Normal'],
                                          fontSize=14, alignment=TA_CENTER)))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                             ParagraphStyle('Date', parent=styles['Normal'],
                                          fontSize=10, alignment=TA_CENTER,
                                          textColor=colors.grey)))

    content.append(PageBreak())

    # Table of Contents
    content.append(Paragraph("Table of Contents", heading1_style))

    toc_data = [
        ["1.", "Overview", "3"],
        ["2.", "Environment Setup", "4"],
        ["3.", "Expert Policy Architecture", "5"],
        ["4.", "Data Collection Pipeline", "6"],
        ["5.", "Data Preprocessing", "8"],
        ["6.", "Quality Control & Validation", "9"],
        ["7.", "Storage & Persistence", "11"],
        ["8.", "Usage Examples", "12"],
        ["9.", "Python Implementation", "14"],
        ["10.", "Troubleshooting", "16"],
    ]

    toc_table = Table(toc_data, colWidths=[0.5*inch, 4*inch, 1*inch])
    toc_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    content.append(toc_table)
    content.append(PageBreak())

    # Chapter 1: Overview
    content.append(Paragraph("1. Overview", heading1_style))

    overview_text = """
    Expert data preparation is crucial for imitation learning and DAgger algorithms. This guide provides
    a comprehensive approach to collecting, processing, and validating expert demonstrations for the
    CarRacing-v2 environment.

    The system consists of four main components:
    """
    content.append(Paragraph(overview_text, styles['Normal']))
    content.append(Spacer(1, 10))

    components_data = [
        ["Component", "Description", "Key Features"],
        ["Expert Policy", "Pre-trained PPO agent", "Discrete→continuous conversion, high performance"],
        ["Data Collection", "Systematic gathering", "Pure demos, DAgger, recovery scenarios"],
        ["Preprocessing", "Frame & action processing", "Normalization, stacking, validation"],
        ["Quality Control", "Validation & analysis", "Integrity checks, anomaly detection, metrics"],
    ]

    components_table = Table(components_data, colWidths=[1.5*inch, 2.5*inch, 2.5*inch])
    components_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(components_table)
    content.append(Spacer(1, 20))

    # Chapter 2: Environment Setup
    content.append(Paragraph("2. Environment Setup", heading1_style))

    setup_text = """
    Before collecting expert data, ensure your environment is properly configured:
    """
    content.append(Paragraph(setup_text, styles['Normal']))

    setup_code = """
    # Activate environment
    conda activate drl-diffdist

    # Verify dependencies
    python -c "import gymnasium, torch, cv2, numpy; print('Dependencies OK')"

    # Check expert model
    ls -la ppo_discrete_carracing.pt
    """
    content.append(Paragraph(setup_code, code_style))

    env_specs = """
    <b>Environment Specifications:</b><br/>
    • Observation Space: RGB images (96,96,3) → Grayscale (84,84,1) → Stacked (4,84,84)<br/>
    • Action Space: Continuous [steer, gas, brake] where steer∈[-1,1], gas/brake∈[0,1]<br/>
    • Frame Processing: Grayscale conversion, resizing, normalization to [0,1]<br/>
    • Episode Length: Variable (typically 200-1000 steps)<br/>
    """
    content.append(Paragraph(env_specs, styles['Normal']))
    content.append(PageBreak())

    # Chapter 3: Expert Policy Architecture
    content.append(Paragraph("3. Expert Policy Architecture", heading1_style))

    expert_text = """
    The expert policy is based on a pre-trained PPO (Proximal Policy Optimization) model with
    the following architecture and characteristics:
    """
    content.append(Paragraph(expert_text, styles['Normal']))

    arch_data = [
        ["Layer", "Input Shape", "Output Shape", "Parameters"],
        ["CNN Feature Extractor", "(4, 84, 84)", "(512,)", "~50K"],
        ["Policy Head", "(512,)", "(5,)", "~2.5K"],
        ["Value Head", "(512,)", "(1,)", "~0.5K"],
        ["Action Converter", "(5,)", "(3,)", "Deterministic"],
    ]

    arch_table = Table(arch_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    content.append(arch_table)
    content.append(Spacer(1, 15))

    action_mapping_code = """
    # Discrete actions (PPO) → Continuous actions (Environment)
    discrete_actions = [
        [-1, 0, 0],    # Turn left
        [1, 0, 0],     # Turn right
        [0, 1, 0],     # Accelerate
        [0, 0, 1],     # Brake
        [0, 0, 0],     # No-op
        # ... additional combinations
    ]
    """
    content.append(Paragraph("Action Mapping:", heading2_style))
    content.append(Paragraph(action_mapping_code, code_style))

    # Chapter 4: Data Collection Pipeline
    content.append(Paragraph("4. Data Collection Pipeline", heading1_style))

    pipeline_text = """
    The data collection system supports three main strategies, each optimized for different
    aspects of imitation learning:
    """
    content.append(Paragraph(pipeline_text, styles['Normal']))

    strategy_data = [
        ["Strategy", "Use Case", "Characteristics", "Volume"],
        ["Pure Expert", "Behavioral Cloning baseline", "Optimal trajectories, high performance", "50-100 episodes"],
        ["DAgger", "Distribution shift correction", "Student attempts + expert labels", "20-30 episodes/iter"],
        ["Recovery", "Robust failure handling", "Off-track, collision scenarios", "10-20% of dataset"],
    ]

    strategy_table = Table(strategy_data, colWidths=[1.5*inch, 2*inch, 2*inch, 1*inch])
    strategy_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(strategy_table)
    content.append(Spacer(1, 15))

    collection_example = """
    # Example: Basic expert collection
    from expert_data_collector import ExpertDataCollector

    collector = ExpertDataCollector("ppo_discrete_carracing.pt")
    stats = collector.collect_expert_episodes(num_episodes=50)
    collector.save_dataset("expert_data.pkl")
    """
    content.append(Paragraph("Basic Collection Example:", heading2_style))
    content.append(Paragraph(collection_example, code_style))
    content.append(PageBreak())

    # Chapter 5: Data Preprocessing
    content.append(Paragraph("5. Data Preprocessing", heading1_style))

    preprocessing_text = """
    Proper preprocessing ensures consistent data format and optimal learning performance.
    The preprocessing pipeline includes observation and action processing steps:
    """
    content.append(Paragraph(preprocessing_text, styles['Normal']))

    content.append(Paragraph("Observation Processing Steps:", heading2_style))
    obs_steps = """
    1. <b>Resize</b>: 96×96 → 84×84 (computational efficiency)<br/>
    2. <b>Grayscale</b>: RGB → single channel (reduces dimensionality)<br/>
    3. <b>Normalization</b>: Pixel values [0,255] → [0,1]<br/>
    4. <b>Frame Stacking</b>: Stack 4 consecutive frames for temporal information<br/>
    5. <b>Channel Reordering</b>: (84,84,4) → (4,84,84) for CNN input
    """
    content.append(Paragraph(obs_steps, styles['Normal']))
    content.append(Spacer(1, 10))

    preprocess_code = """
    def preprocess_obs(obs):
        # Convert (84,84,4) → (4,84,84) with normalization
        return obs.transpose(2,0,1).astype(np.float32) / 255.0
    """
    content.append(Paragraph(preprocess_code, code_style))

    content.append(Paragraph("Action Processing:", heading2_style))
    action_steps = """
    1. <b>Validation</b>: Ensure actions within valid ranges<br/>
    2. <b>Clipping</b>: Constrain to action space bounds<br/>
    3. <b>Normalization</b>: Optional rescaling for training stability
    """
    content.append(Paragraph(action_steps, styles['Normal']))

    # Chapter 6: Quality Control & Validation
    content.append(Paragraph("6. Quality Control & Validation", heading1_style))

    qc_text = """
    Quality control ensures data integrity and expert performance meets requirements.
    The validation system performs comprehensive checks across multiple dimensions:
    """
    content.append(Paragraph(qc_text, styles['Normal']))

    metrics_data = [
        ["Metric Category", "Key Indicators", "Thresholds"],
        ["Expert Performance", "Episode return, completion rate", ">700 avg return, >70% success"],
        ["Data Integrity", "Shape consistency, range validation", "No NaN/inf, correct shapes"],
        ["Action Quality", "Range adherence, smoothness", "Within bounds, low noise"],
        ["Anomaly Detection", "Statistical outliers", "<5% anomaly rate"],
    ]

    metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.red),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(metrics_table)
    content.append(Spacer(1, 15))

    validation_example = """
    # Example: Data validation
    from data_validation_tools import DataValidator

    validator = DataValidator("expert_data.pkl")
    validator.validate_data_integrity()
    validator.analyze_expert_performance()
    validator.create_visualizations("validation_output")
    """
    content.append(Paragraph("Validation Example:", heading2_style))
    content.append(Paragraph(validation_example, code_style))
    content.append(PageBreak())

    # Chapter 7: Storage & Persistence
    content.append(Paragraph("7. Storage & Persistence", heading1_style))

    storage_text = """
    The data management system supports multiple storage formats optimized for different use cases:
    """
    content.append(Paragraph(storage_text, styles['Normal']))

    format_data = [
        ["Format", "Use Case", "Advantages", "File Size"],
        ["Pickle", "Python ecosystem", "Easy serialization, metadata support", "Baseline"],
        ["HDF5", "Large datasets", "Efficient compression, cross-platform", "60-80% of pickle"],
        ["NPZ", "NumPy integration", "Fast loading, compressed", "70-90% of pickle"],
        ["JSON", "Metadata only", "Human readable, lightweight", "5-10% of pickle"],
    ]

    format_table = Table(format_data, colWidths=[1*inch, 2*inch, 2.5*inch, 1*inch])
    format_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(format_table)
    content.append(Spacer(1, 15))

    storage_example = """
    # Example: Data management
    from data_utils import DataManager

    dm = DataManager()
    dataset = dm.load_dataset("expert_data.pkl")

    # Convert formats
    dm.save_dataset(dataset, "data.h5", format="hdf5", compress=True)

    # Split dataset
    train_path, val_path = dm.split_dataset("expert_data.pkl", train_ratio=0.8)

    # Create DataLoader
    dataloader = dm.create_dataloader(dataset, batch_size=32)
    """
    content.append(Paragraph("Data Management Example:", heading2_style))
    content.append(Paragraph(storage_example, code_style))

    # Chapter 8: Usage Examples
    content.append(Paragraph("8. Usage Examples", heading1_style))

    usage_text = """
    Complete workflow examples demonstrating the expert data preparation system:
    """
    content.append(Paragraph(usage_text, styles['Normal']))

    workflow_example = """
    # Complete workflow example

    # 1. Collect expert data
    collector = ExpertDataCollector("ppo_discrete_carracing.pt")
    collector.collect_expert_episodes(num_episodes=100)
    collector.collect_recovery_scenarios(num_episodes=20,
                                       trigger_conditions=['off_track'])
    collector.save_dataset("complete_expert_data.pkl")

    # 2. Validate data quality
    validator = DataValidator("complete_expert_data.pkl")
    validator.validate_data_integrity()
    validator.analyze_expert_performance()
    validator.detect_anomalies(threshold=2.0)
    validator.create_visualizations("validation_results")
    validator.generate_report("quality_report.json")

    # 3. Prepare for training
    dm = DataManager()
    train_path, val_path = dm.split_dataset("complete_expert_data.pkl",
                                           train_ratio=0.8)

    # 4. Create training pipeline
    train_loader = dm.create_dataloader(dm.load_dataset(train_path),
                                       batch_size=64, shuffle=True)
    val_loader = dm.create_dataloader(dm.load_dataset(val_path),
                                     batch_size=64, shuffle=False)

    # 5. Training ready!
    for epoch in range(num_epochs):
        for batch_idx, (obs, actions, rewards) in enumerate(train_loader):
            # Training code here
            pass
    """
    content.append(Paragraph(workflow_example, code_style))
    content.append(PageBreak())

    # Chapter 9: Python Implementation
    content.append(Paragraph("9. Python Implementation", heading1_style))

    impl_text = """
    The implementation consists of three main Python modules, each with specific responsibilities:
    """
    content.append(Paragraph(impl_text, styles['Normal']))

    modules_data = [
        ["Module", "File", "Primary Classes", "Key Functions"],
        ["Data Collection", "expert_data_collector.py", "ExpertDataCollector", "collect_expert_episodes()"],
        ["Validation", "data_validation_tools.py", "DataValidator", "validate_data_integrity()"],
        ["Management", "data_utils.py", "DataManager", "save_dataset(), load_dataset()"],
        ["Examples", "usage_examples.py", "N/A", "Complete workflow demos"],
    ]

    modules_table = Table(modules_data, colWidths=[1.5*inch, 2*inch, 2*inch, 2*inch])
    modules_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.teal),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(modules_table)
    content.append(Spacer(1, 15))

    content.append(Paragraph("Command Line Interface:", heading2_style))
    cli_examples = """
    # Data collection
    python expert_data_collector.py --mode collect --episodes 50 --output expert_data_v1

    # Data validation
    python data_validation_tools.py --input expert_data_v1.pkl --output validation_results

    # Data management
    python data_utils.py --command convert --input data.pkl --output data.h5 --format hdf5

    # Run examples
    python usage_examples.py
    """
    content.append(Paragraph(cli_examples, code_style))

    # Chapter 10: Troubleshooting
    content.append(Paragraph("10. Troubleshooting", heading1_style))

    trouble_text = """
    Common issues and solutions for expert data preparation:
    """
    content.append(Paragraph(trouble_text, styles['Normal']))

    trouble_data = [
        ["Issue", "Symptoms", "Solution"],
        ["Low Expert Performance", "Return < 600, success rate < 70%", "Retrain PPO expert with longer training"],
        ["Action Range Violations", "Actions outside [-1,1] or [0,1]", "Add action clipping and validation"],
        ["Memory Issues", "OOM during collection", "Reduce batch size, use streaming storage"],
        ["Data Corruption", "NaN/inf values", "Check preprocessing pipeline, validate inputs"],
        ["Model Loading Error", "FileNotFoundError", "Verify model path and file permissions"],
    ]

    trouble_table = Table(trouble_data, colWidths=[2*inch, 2.5*inch, 2*inch])
    trouble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(trouble_table)
    content.append(Spacer(1, 15))

    debug_tips = """
    <b>Debugging Tips:</b><br/>
    • Use validation visualizations to inspect data quality<br/>
    • Monitor collection statistics in real-time<br/>
    • Test with small datasets before full collection<br/>
    • Save intermediate results for recovery<br/>
    • Check expert model performance independently
    """
    content.append(Paragraph(debug_tips, styles['Normal']))

    # Conclusion
    content.append(Spacer(1, 20))
    conclusion_text = """
    <b>Conclusion:</b> This guide provides a comprehensive framework for expert data preparation
    in deep reinforcement learning projects. Following these procedures ensures high-quality
    demonstration data that enables successful imitation learning and robust policy performance.
    """
    content.append(Paragraph(conclusion_text, styles['Normal']))

    # Build PDF
    doc.build(content)
    print(f"PDF guide generated: {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    try:
        create_pdf_guide()
        print("\nPDF generation completed successfully!")
        print("The guide includes:")
        print("- Comprehensive methodology")
        print("- Implementation details")
        print("- Code examples")
        print("- Troubleshooting guide")
        print("- Professional formatting")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Ensure all required packages are installed:")
        print("pip install reportlab markdown beautifulsoup4")