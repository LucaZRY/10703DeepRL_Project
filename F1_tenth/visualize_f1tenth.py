"""
F1TENTH Environment Visualization

This script provides visual demonstrations of the F1TENTH environment:
1. Environment rendering with human view
2. LiDAR data visualization
3. Action space demonstration
4. Track layout visualization
5. Vehicle trajectory plotting

Run this to see how the F1TENTH environment looks and behaves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import argparse

# Try to import our wrappers
try:
    from f1tenth_wrappers import make_f1tenth_env, get_f1tenth_maps
except ImportError:
    print("F1TENTH wrappers not found, using basic visualization")

def visualize_lidar_scan(ranges, title="LiDAR Scan", max_range=30.0):
    """
    Visualize a LiDAR scan in polar coordinates.

    Args:
        ranges: Array of range measurements
        title: Plot title
        max_range: Maximum LiDAR range for scaling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Polar plot
    angles = np.linspace(-np.pi, np.pi, len(ranges))

    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.plot(angles, ranges * max_range, 'b-', linewidth=2)
    ax1.fill(angles, ranges * max_range, alpha=0.3)
    ax1.set_ylim(0, max_range)
    ax1.set_title(f"{title} - Polar View")
    ax1.grid(True)

    # Cartesian plot
    ax2.plot(angles, ranges, 'r-', linewidth=2)
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Normalized Range')
    ax2.set_title(f"{title} - Range vs Angle")
    ax2.grid(True)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig

def create_synthetic_lidar_data():
    """Create example LiDAR data to show what F1TENTH observations look like."""

    # Simulate different racing scenarios
    scenarios = {
        'straight_road': "Straight section with side barriers",
        'left_turn': "Approaching left turn",
        'right_turn': "Approaching right turn",
        'chicane': "Complex chicane section",
        'start_finish': "Start/finish straight"
    }

    lidar_data = {}

    for scenario, description in scenarios.items():
        angles = np.linspace(-np.pi, np.pi, 1080)  # F1TENTH standard: 1080 points
        ranges = np.ones_like(angles) * 0.8  # Base distance

        if scenario == 'straight_road':
            # Straight road with side barriers
            front_mask = (angles > -np.pi/6) & (angles < np.pi/6)
            side_mask = (np.abs(angles) > np.pi/2)
            ranges[front_mask] = 0.9 + 0.1 * np.sin(angles[front_mask] * 4)
            ranges[side_mask] = 0.3 + 0.1 * np.random.random(np.sum(side_mask))

        elif scenario == 'left_turn':
            # Left turn - right wall closer
            left_mask = angles < 0
            right_mask = angles > 0
            ranges[left_mask] = 0.8 + 0.2 * np.sin(angles[left_mask] * 2)
            ranges[right_mask] = 0.4 + 0.1 * np.sin(angles[right_mask] * 3)

        elif scenario == 'right_turn':
            # Right turn - left wall closer
            left_mask = angles < 0
            right_mask = angles > 0
            ranges[left_mask] = 0.4 + 0.1 * np.sin(angles[left_mask] * 3)
            ranges[right_mask] = 0.8 + 0.2 * np.sin(angles[right_mask] * 2)

        elif scenario == 'chicane':
            # Complex chicane with varying distances
            ranges = 0.5 + 0.3 * np.sin(angles * 3) + 0.1 * np.sin(angles * 7)
            ranges = np.clip(ranges, 0.2, 1.0)

        elif scenario == 'start_finish':
            # Wide straight with grandstands
            front_mask = (angles > -np.pi/4) & (angles < np.pi/4)
            ranges[front_mask] = 1.0
            ranges[~front_mask] = 0.6 + 0.2 * np.random.random(np.sum(~front_mask))

        # Add noise
        ranges += np.random.normal(0, 0.02, len(ranges))
        ranges = np.clip(ranges, 0.1, 1.0)

        lidar_data[scenario] = {
            'ranges': ranges,
            'description': description
        }

    return lidar_data

def visualize_action_space():
    """Visualize the F1TENTH action space."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Steering angle visualization
    steering_angles = np.linspace(-0.4189, 0.4189, 100)  # ±24 degrees
    steering_degrees = np.degrees(steering_angles)

    ax1.plot(steering_degrees, steering_angles, 'b-', linewidth=3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering Angle (degrees)')
    ax1.set_ylabel('Steering Angle (radians)')
    ax1.set_title('F1TENTH Steering Range: ±24°')
    ax1.grid(True)
    ax1.fill_between(steering_degrees, steering_angles, alpha=0.3)

    # Speed visualization
    speeds = np.linspace(0, 8.0, 100)  # 0-8 m/s
    speeds_kmh = speeds * 3.6  # Convert to km/h

    ax2.plot(speeds, speeds_kmh, 'r-', linewidth=3)
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('F1TENTH Speed Range: 0-8 m/s (0-29 km/h)')
    ax2.grid(True)
    ax2.fill_between(speeds, speeds_kmh, alpha=0.3)

    # Action space 2D visualization
    steering_grid = np.linspace(-0.4189, 0.4189, 20)
    speed_grid = np.linspace(0, 8.0, 20)
    S, V = np.meshgrid(steering_grid, speed_grid)

    # Color by "racing performance" (faster + straighter = better)
    performance = V * (1 - 0.5 * np.abs(S / 0.4189))

    im = ax3.contourf(np.degrees(S), V, performance, levels=15, cmap='viridis')
    ax3.set_xlabel('Steering Angle (degrees)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('F1TENTH Action Space\n(Color = Racing Performance)')
    plt.colorbar(im, ax=ax3, label='Performance Score')

    # Example racing trajectory in action space
    time_steps = np.linspace(0, 10, 100)
    example_steering = 0.2 * np.sin(time_steps) * np.exp(-time_steps/5)
    example_speed = 4 + 2 * np.sin(time_steps/2) * np.exp(-time_steps/8)

    ax4.plot(np.degrees(example_steering), example_speed, 'o-',
             markersize=3, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Steering Angle (degrees)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Example Racing Trajectory')
    ax4.grid(True)
    ax4.set_xlim(-25, 25)
    ax4.set_ylim(0, 8)

    # Add arrows to show direction
    for i in range(0, len(example_steering)-10, 10):
        ax4.annotate('', xy=(np.degrees(example_steering[i+5]), example_speed[i+5]),
                    xytext=(np.degrees(example_steering[i]), example_speed[i]),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

    plt.tight_layout()
    return fig

def visualize_track_layout():
    """Visualize example F1 track layouts."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Define track layouts (simplified)
    tracks = {
        'Spielberg': {
            'corners': [(0.2, 0.3, 'left'), (0.8, 0.7, 'right'), (0.5, 0.9, 'left')],
            'straights': [(0, 0.2), (0.3, 0.8), (0.9, 1.0)],
            'description': 'Austrian GP - Fast with elevation changes'
        },
        'Monaco': {
            'corners': [(0.1, 0.2, 'right'), (0.3, 0.5, 'left'), (0.6, 0.7, 'right'), (0.8, 0.9, 'left')],
            'straights': [(0, 0.1), (0.2, 0.3), (0.5, 0.6)],
            'description': 'Monaco GP - Tight and technical'
        },
        'Monza': {
            'corners': [(0.3, 0.4, 'right'), (0.7, 0.8, 'left')],
            'straights': [(0, 0.3), (0.4, 0.7), (0.8, 1.0)],
            'description': 'Italian GP - High speed circuit'
        },
        'Silverstone': {
            'corners': [(0.2, 0.4, 'right'), (0.5, 0.6, 'left'), (0.7, 0.9, 'right')],
            'straights': [(0, 0.2), (0.4, 0.5), (0.6, 0.7), (0.9, 1.0)],
            'description': 'British GP - Flowing corners'
        },
        'Spa': {
            'corners': [(0.1, 0.3, 'left'), (0.5, 0.7, 'right'), (0.8, 0.9, 'left')],
            'straights': [(0, 0.1), (0.3, 0.5), (0.7, 0.8), (0.9, 1.0)],
            'description': 'Belgian GP - Eau Rouge complex'
        },
        'Suzuka': {
            'corners': [(0.2, 0.3, 'right'), (0.4, 0.6, 'left'), (0.7, 0.8, 'right'), (0.9, 0.95, 'left')],
            'straights': [(0, 0.2), (0.3, 0.4), (0.6, 0.7), (0.8, 0.9)],
            'description': 'Japanese GP - Figure-8 layout'
        }
    }

    for idx, (track_name, track_data) in enumerate(tracks.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Create simplified track layout
        theta = np.linspace(0, 2*np.pi, 1000)
        r = 1.0 + 0.3 * np.sin(6 * theta)  # Base oval shape

        # Add track-specific features
        for corner_start, corner_end, direction in track_data['corners']:
            start_idx = int(corner_start * len(theta))
            end_idx = int(corner_end * len(theta))
            if direction == 'left':
                r[start_idx:end_idx] *= 0.8  # Tighter radius
            else:
                r[start_idx:end_idx] *= 1.2  # Wider radius

        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Plot track
        ax.plot(x, y, 'k-', linewidth=8, alpha=0.8, label='Track')
        ax.fill(x, y, alpha=0.1, color='gray')

        # Add start/finish line
        start_x, start_y = x[0], y[0]
        ax.plot([start_x-0.2, start_x+0.2], [start_y-0.2, start_y+0.2],
                'r-', linewidth=4, label='Start/Finish')

        # Add racing line
        racing_r = r * 0.95  # Slightly inside
        racing_x = racing_r * np.cos(theta)
        racing_y = racing_r * np.sin(theta)
        ax.plot(racing_x, racing_y, 'b--', linewidth=2, alpha=0.7, label='Racing Line')

        ax.set_aspect('equal')
        ax.set_title(f'{track_name}\n{track_data["description"]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    return fig

def run_environment_demo(render_steps=100):
    """Run a live demo of the F1TENTH environment if available."""

    try:
        # Try to create F1TENTH environment
        env = make_f1tenth_env(render_mode="human")
        print("Running F1TENTH environment demo...")

        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        total_reward = 0

        for step in range(render_steps):
            # Random policy for demo
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0:
                print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")

            time.sleep(0.05)  # Slow down for visualization

            if done or truncated:
                print("Episode finished!")
                obs, _ = env.reset()
                total_reward = 0

        env.close()
        return True

    except Exception as e:
        print(f"Could not run environment demo: {e}")
        print("This is normal if F1TENTH gym is not installed.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Visualize F1TENTH environment")
    parser.add_argument("--demo", action="store_true",
                       help="Run live environment demo (requires F1TENTH gym)")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of steps for live demo")
    parser.add_argument("--save", action="store_true",
                       help="Save visualizations to files")

    args = parser.parse_args()

    print("F1TENTH Environment Visualization")
    print("=" * 40)

    # 1. LiDAR Data Visualization
    print("1. Creating LiDAR scan visualizations...")
    lidar_data = create_synthetic_lidar_data()

    for scenario, data in lidar_data.items():
        fig = visualize_lidar_scan(data['ranges'], f"LiDAR: {data['description']}")
        if args.save:
            fig.savefig(f'f1tenth_lidar_{scenario}.png', dpi=150, bbox_inches='tight')
        plt.show(block=False)

    # 2. Action Space Visualization
    print("2. Creating action space visualization...")
    fig_actions = visualize_action_space()
    if args.save:
        fig_actions.savefig('f1tenth_action_space.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)

    # 3. Track Layout Visualization
    print("3. Creating track layout visualizations...")
    fig_tracks = visualize_track_layout()
    if args.save:
        fig_tracks.savefig('f1tenth_tracks.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)

    # 4. Live Demo (optional)
    if args.demo:
        print("4. Running live environment demo...")
        success = run_environment_demo(args.steps)
        if not success:
            print("Live demo not available. Install F1TENTH gym for interactive visualization.")

    print("\nVisualization complete!")
    print("Available F1 tracks:", get_f1tenth_maps())

    # Keep plots open
    plt.show()

if __name__ == "__main__":
    main()