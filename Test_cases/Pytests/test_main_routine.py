import sys
import pytest
from unittest.mock import MagicMock

import Hydrological_model_validator.__main__ as main_script

################################################################################
# ---------- Tests for main ----------
################################################################################

# Test the normal execution path of main()
# It should parse arguments, print the banner, call generate_full_report once,
# and print the output directory path.
def test_main_runs_and_calls_report(monkeypatch, capsys):
    # Patch generate_full_report *inside* the __main__ module
    mock_generate = MagicMock()
    monkeypatch.setattr("Hydrological_model_validator.Report_generator.generate_full_report", mock_generate)
    # Patch print_banner
    monkeypatch.setattr("Hydrological_model_validator.__main__.print_banner", MagicMock())
    # Set argv with required input argument
    monkeypatch.setattr(sys, "argv", ["progname", "data_folder"])
    # Run main
    import Hydrological_model_validator.__main__ as main_script
    main_script.main()

    # Assert banner printed once
    main_script.print_banner.assert_called_once()
    # Assert generate_full_report was called once
    mock_generate.assert_called_once()

    # Check stdout for output directory message
    captured = capsys.readouterr()
    assert "Output saved in:" in captured.out


# Test that using the --no-banner flag suppresses printing the banner.
def test_main_no_banner(monkeypatch, capsys):
    # Patch generate_full_report to no-op to avoid side effects.
    monkeypatch.setattr(
        "Hydrological_model_validator.Report_generator.generate_full_report",
        lambda **kwargs: None
    )
    # Patch print_banner to raise error if called, so test fails if banner prints.
    monkeypatch.setattr(main_script, "print_banner", lambda: pytest.fail("Banner should not print"))
    # Simulate CLI args including --no-banner flag.
    monkeypatch.setattr(sys, "argv", ["progname", "data_folder", "--no-banner"])
    # Run main, which should NOT print the banner.
    main_script.main()
    captured = capsys.readouterr()
    # Confirm output directory message still prints.
    assert "Output saved in:" in captured.out


# Test that the --info flag prints program info and exits with status 0.
def test_main_info_flag_exits(monkeypatch, capsys):
    # Simulate CLI args including --info flag.
    monkeypatch.setattr(sys, "argv", ["progname", "dummy", "--info"])
    # Expect SystemExit because main calls sys.exit(0) after printing info.
    with pytest.raises(SystemExit) as e:
        main_script.main()
    captured = capsys.readouterr()
    # Confirm the printed output includes the version string.
    assert "Hydrological Model Validator - Preliminary Report Generator" in captured.out
    # Confirm exit code is zero (normal exit).
    assert e.value.code == 0


# Test that input argument can be a valid dict-like string.
def test_main_input_as_dict(monkeypatch):
    import Hydrological_model_validator.__main__ as main_script  # import inside function to avoid circular imports
    # Patch generate_full_report to a MagicMock on the correct module path.
    mock_generate = MagicMock()
    monkeypatch.setattr(
        "Hydrological_model_validator.Report_generator.generate_full_report",
        mock_generate
    )
    # Patch print_banner to confirm it gets called.
    monkeypatch.setattr(main_script, "print_banner", MagicMock())
    # Provide CLI arg with a JSON-like dict string.
    dict_input = '{"file1": "path/to/file1.nc", "file2": "path/to/file2.nc"}'
    monkeypatch.setattr(sys, "argv", ["progname", dict_input])
    # Run main, which should parse dict input and call report generator.
    main_script.main()
    # Banner should be printed.
    main_script.print_banner.assert_called_once()
    # Report generator should be called once.
    mock_generate.assert_called_once()


# Test to verify that the code exits with invalid inputs
def test_main_invalid_dict_input(monkeypatch):
    # Provide malformed dict string as input (must start and end with braces to trigger dict parsing)
    monkeypatch.setattr(sys, "argv", ["progname", "{invalid_dict}"])  # <-- Note closing brace
    # Patch generate_full_report to avoid side effects (not needed here if sys.exit works)
    monkeypatch.setattr(
        "Hydrological_model_validator.Report_generator.generate_full_report",
        lambda *args, **kwargs: None
    )
    # Run main, expecting it to raise SystemExit due to failed dict parsing.
    with pytest.raises(SystemExit):
        main_script.main()


# Test that missing required input argument causes argument parsing error.
def test_main_missing_input(monkeypatch):
    # Simulate CLI args without the required 'input' argument.
    monkeypatch.setattr(sys, "argv", ["progname"])
    # main() should raise SystemExit due to missing required input.
    with pytest.raises(SystemExit):
        main_script.main()


# Test that --verbose flag triggers verbose print statements during execution.
def test_main_verbose_flag(monkeypatch, capsys):
    # Patch generate_full_report to MagicMock to avoid real execution.
    monkeypatch.setattr(
        "Hydrological_model_validator.Report_generator.generate_full_report",
        MagicMock()
    )
    # Patch print_banner to allow banner print.
    monkeypatch.setattr(main_script, "print_banner", MagicMock())
    # Simulate CLI args including --verbose flag.
    monkeypatch.setattr(sys, "argv", ["progname", "data_folder", "--verbose"])
    
    main_script.main()
    captured = capsys.readouterr()
    # Confirm that verbose "Starting report generation" message is printed.
    assert "Starting report generation" in captured.out


# Test that all other flags are passed correctly to generate_full_report.
def test_main_other_flags(monkeypatch):
    # Patch generate_full_report to MagicMock at its original module location.
    mock_generate = MagicMock()
    monkeypatch.setattr(
        "Hydrological_model_validator.Report_generator.generate_full_report",
        mock_generate
    )
    # Patch print_banner to allow banner print.
    monkeypatch.setattr(main_script, "print_banner", MagicMock())
    # Simulate CLI args with multiple flags and options.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "progname",
            "data_folder",
            "--no-pdf",
            "--check",
            "--open-report",
            "--variable", "precip",
            "--unit", "mm/h"
        ]
    )
    main_script.main()
    # Extract kwargs passed to generate_full_report.
    called_kwargs = mock_generate.call_args[1]
    # Verify flag values passed correctly.
    assert called_kwargs["generate_pdf"] is False
    assert called_kwargs["check_only"] is True
    assert called_kwargs["open_report"] is True
    assert called_kwargs["variable"] == "precip"
    assert called_kwargs["unit"] == "mm/h"