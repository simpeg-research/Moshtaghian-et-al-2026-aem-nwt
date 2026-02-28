# main.py

from run_aem_inversion import execute_inversion_workflow

if __name__ == "__main__":
    # Select AEM line and configure execution flags
    selected_line = "L150020"  # Options: L120030, L150020, L160030

    execute_inversion_workflow(
        selected_line=selected_line,
        data_file="data/AEM_NWT_PaperLines.txt",
        output_dir="outputs",
        run_inversion=False,
        save_results=True,
        run_plotting=True,
        model_to_plot="fixedbeta",  # options: "fixedbeta", "avg"
        plot_doi_mode="none",       # options: "none", "oldenburg", "christiansen"
        compute_pbtz=True,
        pbtz_p=0.50,
        pbtz_full_width_min=2.0,
        pbtz_full_width_max=10.0,
    )