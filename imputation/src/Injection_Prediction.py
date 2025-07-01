import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from InjPred_Train import PowerSystemInjectionPredictor as InjPred

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_array_comparison(array1, array2, array_names=['Array 1', 'Array 2'],
                          bus_names=None, title_prefix="Array Comparison"):
    """
    Comprehensive plotting and comparison of two arrays with shape (1, 132, 3)
    """

    # Reshape arrays from (1, 132, 3) to (132, 3)
    arr1 = array1.reshape(132, 3)
    arr2 = array2.reshape(132, 3)

    # Generate bus names if not provided
    if bus_names is None:
        bus_names = [f'Bus_{i + 1}' for i in range(132)]

    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    # Color scheme for phases
    phase_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    phase_names = ['Phase 1', 'Phase 2', 'Phase 3']

    # 1. Line plots for each phase (Top row)
    for phase in range(3):
        ax = fig.add_subplot(gs[0, phase])

        # Plot both arrays
        ax.plot(range(132), arr1[:, phase], 'o-', color=phase_colors[phase],
                alpha=0.7, linewidth=2, markersize=4, label=array_names[0])
        ax.plot(range(132), arr2[:, phase], 's--', color=phase_colors[phase],
                alpha=0.9, linewidth=2, markersize=3, label=array_names[1])

        ax.set_title(f'{phase_names[phase]} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Bus Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight significant differences
        diff = np.abs(arr1[:, phase] - arr2[:, phase])
        threshold = np.std(diff) * 2  # 2 standard deviations
        significant_diff_indices = np.where(diff > threshold)[0]

        for idx in significant_diff_indices:
            ax.axvline(x=idx, color='red', alpha=0.3, linestyle=':', linewidth=1)

    # # 2. Scatter plots (Middle row)
    # for phase in range(3):
    #     ax = fig.add_subplot(gs[1, phase])
    #
    #     # Create scatter plot
    #     scatter = ax.scatter(arr1[:, phase], arr2[:, phase],
    #                          c=range(132), cmap='viridis', alpha=0.7, s=50)
    #
    #     # Perfect correlation line
    #     min_val = min(np.min(arr1[:, phase]), np.min(arr2[:, phase]))
    #     max_val = max(np.max(arr1[:, phase]), np.max(arr2[:, phase]))
    #     ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    #
    #     # Calculate correlation
    #     correlation = np.corrcoef(arr1[:, phase], arr2[:, phase])[0, 1]
    #
    #     ax.set_title(f'{phase_names[phase]} Correlation\n(r = {correlation:.3f})',
    #                  fontsize=12, fontweight='bold')
    #     ax.set_xlabel(f'{array_names[0]} Values')
    #     ax.set_ylabel(f'{array_names[1]} Values')
    #     ax.grid(True, alpha=0.3)
    #
    #     # Add colorbar
    #     plt.colorbar(scatter, ax=ax, label='Bus Index')
    #
    # # 3. Difference analysis (Bottom row)
    #
    # # 3a. Difference heatmap
    # ax = fig.add_subplot(gs[2, 0])
    # differences = arr1 - arr2
    #
    # im = ax.imshow(differences.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    # ax.set_title('Difference Heatmap\n(Array1 - Array2)', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Bus Index')
    # ax.set_ylabel('Phase')
    # ax.set_yticks([0, 1, 2])
    # ax.set_yticklabels(['Phase 1', 'Phase 2', 'Phase 3'])
    # plt.colorbar(im, ax=ax, label='Difference')
    #
    # # 3b. Overall difference histogram
    # ax = fig.add_subplot(gs[2, 1])
    # all_diffs = differences.flatten()
    # ax.hist(all_diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    # ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    # ax.set_title('Difference Distribution', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Difference (Array1 - Array2)')
    # ax.set_ylabel('Frequency')
    # ax.grid(True, alpha=0.3)
    #
    # # Add statistics text
    # mean_diff = np.mean(all_diffs)
    # std_diff = np.std(all_diffs)
    # ax.text(0.02, 0.98, f'Mean: {mean_diff:.4f}\nStd: {std_diff:.4f}',
    #         transform=ax.transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #
    # # 3c. Box plot comparison
    # ax = fig.add_subplot(gs[2, 2])
    #
    # # Prepare data for box plot
    # box_data = []
    # box_labels = []
    # for phase in range(3):
    #     box_data.extend([arr1[:, phase], arr2[:, phase]])
    #     box_labels.extend([f'{array_names[0]}\n{phase_names[phase]}',
    #                        f'{array_names[1]}\n{phase_names[phase]}'])
    #
    # bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    #
    # # Color the boxes
    # colors = []
    # for phase in range(3):
    #     colors.extend([phase_colors[phase], phase_colors[phase]])
    #
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
    #     patch.set_alpha(0.7)
    #
    # ax.set_title('Value Distribution by Phase', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Value')
    # plt.xticks(rotation=45)
    # ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix} - Shape: {array1.shape}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


def detailed_comparison_stats(array1, array2, array_names=['Array 1', 'Array 2']):
    """
    Print detailed statistical comparison
    """
    arr1 = array1.reshape(132, 3)
    arr2 = array2.reshape(132, 3)

    print("=" * 80)
    print("DETAILED COMPARISON STATISTICS")
    print("=" * 80)

    print(f"Array shapes: {array1.shape}")
    print(f"Comparison between: {array_names[0]} vs {array_names[1]}")

    phase_names = ['Phase 1', 'Phase 2', 'Phase 3']

    for phase in range(3):
        print(f"\n{phase_names[phase]}:")
        print("-" * 40)

        # Basic statistics
        print(f"{array_names[0]:15} - Mean: {np.mean(arr1[:, phase]):8.4f}, Std: {np.std(arr1[:, phase]):8.4f}")
        print(f"{array_names[1]:15} - Mean: {np.mean(arr2[:, phase]):8.4f}, Std: {np.std(arr2[:, phase]):8.4f}")

        # Differences
        diff = arr1[:, phase] - arr2[:, phase]
        abs_diff = np.abs(diff)

        print(f"{'Difference':15} - Mean: {np.mean(diff):8.4f}, Std: {np.std(diff):8.4f}")
        print(f"{'Abs Difference':15} - Mean: {np.mean(abs_diff):8.4f}, Max: {np.max(abs_diff):8.4f}")

        # Correlation
        correlation = np.corrcoef(arr1[:, phase], arr2[:, phase])[0, 1]
        print(f"{'Correlation':15} - r = {correlation:.6f}")

        # Percentage differences
        percent_diff = np.abs(diff) / np.maximum(np.abs(arr1[:, phase]), 1e-10) * 100
        print(f"{'% Difference':15} - Mean: {np.mean(percent_diff):8.2f}%, Max: {np.max(percent_diff):8.2f}%")

    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print("-" * 40)
    all_diff = (arr1 - arr2).flatten()
    print(f"Total elements compared: {len(all_diff)}")
    print(f"Overall mean difference: {np.mean(all_diff):8.4f}")
    print(f"Overall std difference: {np.std(all_diff):8.4f}")
    print(f"Overall max abs difference: {np.max(np.abs(all_diff)):8.4f}")


def create_detailed_plots_plotly(pred, measurement, array_names=['Predicted', 'Measured']):
    """
    Create interactive detailed plots using Plotly
    """
    bus = pred.shape[1]
    ph = pred.shape[2]
    arr_pred = pred.reshape(bus, ph)
    arr_measurement = measurement.reshape(bus, ph)

    phase_colors = ['#E74C3C', '#27AE60', '#3498DB']
    phase_names = ['Phase A', 'Phase B', 'Phase C']

    print("=== FILTERING SUMMARY ===")

    # Create subplots with secondary y-axes
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=phase_names,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]]
    )

    for phase in range(ph):
        row = phase + 1

        # Get data for this phase
        pred_phase = arr_pred[:, phase]
        measurement_phase = arr_measurement[:, phase]

        # Create mask for values >= 0.2 in predictions
        mask_show = pred_phase >= 0.2
        total_points = len(pred_phase)
        shown_points = np.sum(mask_show)

        x_indices = np.arange(bus)

        # Only plot markers where pred >= 0.2
        if np.any(mask_show):
            x_shown = x_indices[mask_show]
            pred_shown = pred_phase[mask_show]
            measurement_shown = measurement_phase[mask_show]

            pred_all = pred_phase
            measurement_all = measurement_phase

            # Calculate statistics
            rmse = np.sqrt(np.mean((pred_all - measurement_all) ** 2))
            mae = np.mean(np.abs(pred_all - measurement_all))

            # Add prediction trace with circles
            fig.add_trace(
                go.Scatter(
                    x=x_shown,
                    y=pred_shown,
                    mode='lines+markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=phase_colors[phase],
                        line=dict(color='white', width=2)
                    ),
                    line=dict(color=phase_colors[phase], width=3),
                    name=f'Prediction - {phase_names[phase]}',
                    legendgroup=f'group{phase}',
                    hovertemplate='<b>Prediction</b><br>' +
                                  'Bus: %{x}<br>' +
                                  'Value: %{y:.3f} kW<br>' +
                                  '<extra></extra>',
                    showlegend=True
                ),
                row=row, col=1, secondary_y=False
            )

            # Add measurement trace with triangles
            fig.add_trace(
                go.Scatter(
                    x=x_shown,
                    y=measurement_shown,
                    mode='lines+markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='white',
                        line=dict(color=phase_colors[phase], width=3)
                    ),
                    line=dict(color=phase_colors[phase], width=3, dash='dash'),
                    name=f'Measurement - {phase_names[phase]}',
                    legendgroup=f'group{phase}',
                    hovertemplate='<b>Measurement</b><br>' +
                                  'Bus: %{x}<br>' +
                                  'Value: %{y:.3f} kW<br>' +
                                  '<extra></extra>',
                    showlegend=True
                ),
                row=row, col=1, secondary_y=False
            )

            # Calculate differences and add difference bars
            differences = pred_phase[mask_show] - measurement_phase[mask_show]
            max_diff = np.max(np.abs(differences))

            # Color bars by difference magnitude
            bar_colors = ['red' if abs(d) > np.std(differences) else 'lightgray'
                          for d in differences]

            # Add difference bars (top-down effect)
            fig.add_trace(
                go.Bar(
                    x=x_shown,
                    y=differences,
                    base=max_diff,  # Start bars from max_diff
                    marker=dict(
                        color=bar_colors,
                        opacity=0.6,
                        line=dict(color='rgba(0,0,0,0.3)', width=1)
                    ),
                    name=f'Difference - {phase_names[phase]}',
                    legendgroup=f'diff{phase}',
                    hovertemplate='<b>Difference</b><br>' +
                                  'Bus: %{x}<br>' +
                                  'Diff: %{y:.3f} kW<br>' +
                                  'Pred - Measured<br>' +
                                  '<extra></extra>',
                    showlegend=True,
                    yaxis='y2'
                ),
                row=row, col=1, secondary_y=True
            )

            # Set y-axis limits for main plot
            max_valInjection = int(np.ceil(max(np.max(pred_shown), np.max(measurement_shown))))
            fig.update_yaxes(
                range=[0, max_valInjection + 3],
                title_text='DER Injection (kW)',
                row=row, col=1, secondary_y=False
            )

            # Set y-axis limits for difference plot (top-down effect)
            fig.update_yaxes(
                range=[max_diff + np.min(differences), max_diff + np.max(differences)],
                title_text='Difference kW (Top-down)',
                title_font_color='gray',
                tickfont_color='gray',
                row=row, col=1, secondary_y=True
            )

        else:
            # Add text for no data - using paper coordinates
            fig.add_annotation(
                text=f'No prediction values ≥ 0.2 in {phase_names[phase]}',
                x=0.5, y=0.8 - (phase * 0.33),  # Adjust y position for each subplot
                xref='paper', yref='paper',
                showarrow=False,
                bgcolor='lightgray',
                font=dict(size=12, color='black'),
                bordercolor='gray',
                borderwidth=1
            )

        # Update x-axis
        fig.update_xaxes(
            title_text='Bus Index' if phase == ph - 1 else '',
            range=[-1, bus + 1],
            dtick=1 if shown_points <= 20 else max(1, shown_points // 10),
            row=row, col=1
        )

    # Add statistics annotations after the loop (using paper coordinates)
    for phase in range(ph):
        pred_phase = arr_pred[:, phase]
        measurement_phase = arr_measurement[:, phase]
        mask_show = pred_phase >= 0.2
        total_points = len(pred_phase)
        shown_points = np.sum(mask_show)

        if np.any(mask_show):
            pred_all = pred_phase
            measurement_all = measurement_phase
            rmse = np.sqrt(np.mean((pred_all - measurement_all) ** 2))
            mae = np.mean(np.abs(pred_all - measurement_all))

            fig.add_annotation(
                text=f'Points: {shown_points}/{total_points}<br>RMSE: {rmse:.3f}<br>MAE: {mae:.3f}',
                x=0.02,
                y=0.95 - (phase * 0.33),  # Position for each subplot
                xref='paper',
                yref='paper',
                showarrow=False,
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=10),
                xanchor='left',
                yanchor='top'
            )

    # Update layout for better interactivity
    fig.update_layout(
        title=dict(
            text='Interactive Phase-by-Phase Comparison<br><sub>Hover for details, click legend to toggle</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        height=1000,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        template='plotly_white',
        showlegend=True
    )

    # Add range slider to bottom subplot
    # fig.update_xaxes(rangeslider=dict(visible=True), row=3, col=1)
    fig.show()

    return fig


def create_detailed_plots(pred, measurement, array_names=['Predicted', 'Measured']):
    """
    Create additional detailed plots
    """
    bus = pred.shape[1]
    ph = pred.shape[2]
    arr_pred = pred.reshape(bus, ph)
    arr_measurement = measurement.reshape(bus, ph)

    # 1. Individual phase comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    phase_colors = ['#E74C3C', '#27AE60', '#3498DB']
    phase_names = ['Phase A', 'Phase B', 'Phase C']

    print("=== FILTERING SUMMARY ===")

    for phase in range(ph):
        ax = axes[phase]

        # Get data for this phase
        pred_phase = arr_pred[:, phase]
        measurement_phase = arr_measurement[:, phase]

        # Create mask for values >= 0.2 in predictions
        mask_show = pred_phase >= 0.2
        total_points = len(pred_phase)
        shown_points = np.sum(mask_show)

        x_indices = np.arange(bus)

        # Only plot markers where pred >= 0.2
        if np.any(mask_show):
            x_shown = x_indices[mask_show]
            pred_shown = pred_phase[mask_show]
            measurement_shown = measurement_phase[mask_show]

            pred_all = pred_phase
            measurement_all = measurement_phase

            # Plot prediction with circle markers
            ax.plot(x_shown, pred_shown, 'o-', color=phase_colors[phase],
                    alpha=0.8, linewidth=2, markersize=6, label='Prediction',
                    markerfacecolor=phase_colors[phase], markeredgecolor='white',
                    markeredgewidth=1)

            # Plot measurement with triangle markers
            ax.plot(x_shown, measurement_shown, '^-', color=phase_colors[phase],
                    alpha=0.9, linewidth=2, markersize=8, label='Measurement',
                    markerfacecolor='white', markeredgecolor=phase_colors[phase],
                    markeredgewidth=2)

            # REMOVED: Fill between (colored background)
            # ax.fill_between(x_shown, pred_shown, measurement_shown,
            #                 alpha=0.2, color=phase_colors[phase])

            # Calculate and display statistics for shown points
            rmse = np.sqrt(np.mean((pred_shown - measurement_shown) ** 2))
            mae = np.mean(np.abs(pred_shown - measurement_shown))
            # rmse = np.sqrt(np.mean((pred_all - measurement_all) ** 2))
            # mae = np.mean(np.abs(pred_all - measurement_all))

            # ax.text(0.02, 0.98, f'Points shown: {shown_points}/{total_points}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}',
            #         transform=ax.transAxes, verticalalignment='top',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            #         fontsize=10)

        else:
            ax.text(0.5, 0.5, f'No prediction values ≥ 0.2 in {phase_names[phase]}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        ax.set_title(f'{phase_names[phase]} - Filtered Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Bus Index')
        ax.set_ylabel('DER Injection (kW)')
        if np.any(mask_show):
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Set x-axis to show actual indices (not continuous range)
        if np.any(mask_show):
            x_shown = x_indices[mask_show]
            ax.set_xticks(x_shown)
            ax.set_xticklabels(x_shown)
        ax.set_xlim(-1, bus+1)
        max_valInjection = int(np.ceil(max(np.max(pred_shown), np.max(measurement_shown))) )
        ax.set_ylim(0, max_valInjection+3)

        # Add difference subplot
        ax2 = ax.twinx()
        if np.any(mask_show):
            x_shown = x_indices[mask_show]
            differences = pred_phase[mask_show] - measurement_phase[mask_show]

            # Color bars by difference magnitude
            colors = ['red' if abs(d) > np.std(differences) else 'lightgray'
                      for d in differences]


            # Get the maximum difference for reference
            max_diff = np.max(np.abs(differences))
            print('max_diff', max_diff)
            # Create bars that start from top and go down
            bars = ax2.bar(x_shown, differences, alpha=0.4, color=colors, width=0.8,
                           bottom=max_diff)  # Start bars from top

            # Set limits to show the flipped effect
            ax2.set_ylim(max_diff + np.max(differences), max_diff + np.min(differences))


        ax2.set_ylabel('Difference kW (Top-down)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    plt.savefig("../output/Imputed_data_Detection.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig

def create_detailed_plots_allNodes(pred, measurement, array_names=['Predicted', 'Measured']):
    """
    Create additional detailed plots
    """
    array1 = pred
    array2 = measurement
    bus = array1.shape[1]
    ph = array1.shape[2]
    arr1 = array1.reshape(bus, ph)
    arr2 = array2.reshape(bus, ph)

    # 1. Individual phase comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    phase_colors = ['red', 'green', 'blue']
    phase_names = ['Phase A', 'Phase B', 'Phase C']

    for phase in range(3):
        ax = axes[phase]

        # Plot both arrays
        x_indices = range(132)
        ax.plot(x_indices, arr1[:, phase], 'o', color=phase_colors[phase],
                alpha=0.7, linewidth=2, markersize=4, label=array_names[0])
        ax.plot(x_indices, arr2[:, phase], 's', color=phase_colors[phase],
                alpha=0.9, linewidth=2, markersize=7, label=array_names[1])

        # Fill between to show differences
        ax.fill_between(x_indices, arr1[:, phase], arr2[:, phase],
                        alpha=0.3, color=phase_colors[phase])

        ax.set_title(f'{phase_names[phase]} - Detailed Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Bus Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add difference subplot
        ax2 = ax.twinx()
        differences = arr1[:, phase] - arr2[:, phase]
        ax2.bar(x_indices, differences, alpha=0.3, color='gray', width=0.8)
        ax2.set_ylabel('Difference', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    plt.suptitle('Phase-by-Phase Detailed Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


    return fig


def grid_measurements(sample = 10):
    # baseKV = 1
    # baseS = 1
    # baseZ = baseKV ** 2 / baseS
    file_path = "../training_data/"

    voltage_data = np.load(file_path+'voltage.npy')
    voltage_data_t = voltage_data[sample, :, :]

    base_r = np.load(file_path + 'r.npy')
    base_x = np.load(file_path + 'x.npy')

    load_forecast_data = np.load(file_path+'load.npy')
    load_forecast_data_t = load_forecast_data[sample, :, :]

    injection_data = np.load(file_path+'injection.npy')
    injection_data_t = injection_data[sample, :, :]

    voltage_data_t = np.tile(voltage_data_t, (1, 1, 1))
    load_forecast_data_t = np.tile(load_forecast_data_t, (1, 1, 1))
    r_data = np.tile(base_r, (1, 1, 1, 1))
    x_data = np.tile(base_x, (1, 1, 1, 1))
    injection_data_t = np.tile(injection_data_t, (1, 1, 1))

    n_buses = voltage_data.shape[1]
    n_phases = voltage_data.shape[2]

    return n_buses, n_phases, voltage_data_t, load_forecast_data_t, r_data, x_data, injection_data_t


if __name__ == "__main__":

    n_buses, n_phases, voltage_data, load_forecast_data, r_data, x_data, injection_data = grid_measurements(sample = 52)
    injection_data[0, 69, :] = [0, 0, 0]
    injection_data[0, 74, :] = [0, 0, 0]
    agentFilePath = "power_system_injection_predictor.h5"

    predictor = InjPred(n_buses, n_phases)
    predictor.load_model(agentFilePath)
    predicted_injections = predictor.predict(voltage_data, load_forecast_data, r_data, x_data)

    # Additional detailed plots
    fig2 = create_detailed_plots(predicted_injections, injection_data,
                                 array_names=['Predicted', 'Actual'])
