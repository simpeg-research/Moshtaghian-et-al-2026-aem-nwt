# main.py

from run_aem_inversion import execute_inversion_workflow

if __name__ == "__main__":
    # Select AEM line and configure execution flags
    selected_line = "L120030"  # Options: L120030, L150020, L160030

    execute_inversion_workflow(
        selected_line,
        data_file='data/AEM_NWT_PaperLines.txt',
        output_dir='outputs',
        run_inversion=True,
        save_results=True,
        run_plotting=True,
        model_to_plot="fixedbeta",  # options: "fixedbeta", "avg"
        plot_doi_mode="none", # Options: "none", "oldenburg", "christiansen",
        line_name=selected_line
    )