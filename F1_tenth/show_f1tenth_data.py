"""
Simple F1TENTH Data Visualization

Shows what F1TENTH observations and actions look like without requiring GUI.
Generates plots that demonstrate the environment characteristics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def create_sample_lidar_data():
    """Create realistic F1TENTH LiDAR data samples."""

    # Standard F1TENTH LiDAR: 1080 points, ±135° FOV
    num_points = 1080
    fov = 270  # degrees total field of view
    angles = np.linspace(-np.radians(fov/2), np.radians(fov/2), num_points)

    scenarios = {}

    # 1. Straight corridor
    ranges_straight = np.ones(num_points) * 15.0  # 15m base distance
    # Side walls at ~3m
    side_mask = (np.abs(angles) > np.radians(60))
    ranges_straight[side_mask] = 3.0
    # Front open road
    front_mask = (np.abs(angles) < np.radians(30))
    ranges_straight[front_mask] = 20.0
    # Add some noise
    ranges_straight += np.random.normal(0, 0.5, num_points)
    ranges_straight = np.clip(ranges_straight, 0.5, 30.0)
    scenarios['straight'] = ranges_straight

    # 2. Left turn approaching
    ranges_left = np.ones(num_points) * 8.0
    # Right wall closer
    right_mask = (angles > 0)
    ranges_left[right_mask] = 2.0 + 3.0 * np.sin(angles[right_mask] * 2)**2
    # Left wall further
    left_mask = (angles < 0)
    ranges_left[left_mask] = 8.0 + 5.0 * np.cos(angles[left_mask])**2
    ranges_left += np.random.normal(0, 0.3, num_points)
    ranges_left = np.clip(ranges_left, 0.5, 30.0)
    scenarios['left_turn'] = ranges_left

    # 3. Right turn approaching
    ranges_right = np.ones(num_points) * 8.0
    # Left wall closer
    left_mask = (angles < 0)
    ranges_right[left_mask] = 2.0 + 3.0 * np.sin(angles[left_mask] * 2)**2
    # Right wall further
    right_mask = (angles > 0)
    ranges_right[right_mask] = 8.0 + 5.0 * np.cos(angles[right_mask])**2
    ranges_right += np.random.normal(0, 0.3, num_points)
    ranges_right = np.clip(ranges_right, 0.5, 30.0)
    scenarios['right_turn'] = ranges_right

    return angles, scenarios

def create_sample_actions():
    """Create sample F1TENTH action trajectories."""

    time_steps = np.linspace(0, 10, 200)  # 10 seconds at 20Hz

    trajectories = {}

    # 1. Straight line racing
    steering_straight = 0.05 * np.sin(time_steps * 0.5)  # Small corrections
    speed_straight = 6.0 + 1.0 * np.sin(time_steps * 0.3)  # Varying speed
    trajectories['straight'] = (steering_straight, speed_straight)

    # 2. Left turn sequence
    steering_left = -0.3 * np.exp(-(time_steps - 5)**2 / 4)  # Turn at t=5s
    speed_left = 5.0 - 2.0 * np.abs(steering_left) / 0.3  # Slow for turns
    trajectories['left_turn'] = (steering_left, speed_left)

    # 3. Chicane (S-turns)
    steering_chicane = 0.2 * np.sin(time_steps * 2)  # Quick alternating turns
    speed_chicane = 4.0 + 1.0 * np.cos(time_steps * 1.5)  # Variable speed
    trajectories['chicane'] = (steering_chicane, speed_chicane)

    return time_steps, trajectories

def plot_lidar_scenarios():
    """Create LiDAR visualization plots."""

    angles, scenarios = create_sample_lidar_data()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (name, ranges) in enumerate(scenarios.items()):
        # Polar plot
        ax_polar = plt.subplot(2, 3, idx + 1, projection='polar')
        ax_polar.plot(angles, ranges, 'b-', linewidth=2, alpha=0.8)
        ax_polar.fill(angles, ranges, alpha=0.2, color='blue')
        ax_polar.set_ylim(0, 30)
        ax_polar.set_title(f'{name.replace("_", " ").title()}\nLiDAR Scan (Polar)')
        ax_polar.grid(True, alpha=0.3)

        # Cartesian plot
        ax_cart = plt.subplot(2, 3, idx + 4)
        ax_cart.plot(np.degrees(angles), ranges, 'r-', linewidth=2)
        ax_cart.fill_between(np.degrees(angles), ranges, alpha=0.2, color='red')
        ax_cart.set_xlabel('Angle (degrees)')
        ax_cart.set_ylabel('Range (meters)')
        ax_cart.set_title(f'{name.replace("_", " ").title()}\nRange vs Angle')
        ax_cart.grid(True, alpha=0.3)
        ax_cart.set_xlim(-135, 135)
        ax_cart.set_ylim(0, 25)

    plt.tight_layout()
    plt.savefig('f1tenth_lidar_scenarios.png', dpi=150, bbox_inches='tight')
    print("Saved: f1tenth_lidar_scenarios.png")
    return fig

def plot_action_trajectories():
    """Create action space visualization plots."""

    time_steps, trajectories = create_sample_actions()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot all steering trajectories
    for name, (steering, speed) in trajectories.items():
        ax1.plot(time_steps, np.degrees(steering), linewidth=2, label=name.replace('_', ' ').title())

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Steering Angle (degrees)')
    ax1.set_title('F1TENTH Steering Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-25, 25)

    # Plot all speed trajectories
    for name, (steering, speed) in trajectories.items():
        ax2.plot(time_steps, speed, linewidth=2, label=name.replace('_', ' ').title())

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('F1TENTH Speed Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 8)

    # Action space scatter plot
    all_steering = []
    all_speed = []
    colors = []
    color_map = {'straight': 'blue', 'left_turn': 'red', 'chicane': 'green'}

    for name, (steering, speed) in trajectories.items():
        all_steering.extend(steering)
        all_speed.extend(speed)
        colors.extend([color_map[name]] * len(steering))

    scatter = ax3.scatter(np.degrees(all_steering), all_speed, c=colors, alpha=0.6, s=20)
    ax3.set_xlabel('Steering Angle (degrees)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('F1TENTH Action Space Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-25, 25)
    ax3.set_ylim(0, 8)

    # Add action space boundaries
    steering_range = [-24, 24]  # degrees
    speed_range = [0, 8]  # m/s
    ax3.axvline(steering_range[0], color='black', linestyle='--', alpha=0.5, label='Limits')
    ax3.axvline(steering_range[1], color='black', linestyle='--', alpha=0.5)
    ax3.axhline(speed_range[0], color='black', linestyle='--', alpha=0.5)
    ax3.axhline(speed_range[1], color='black', linestyle='--', alpha=0.5)

    # Speed vs steering correlation
    example_steering, example_speed = trajectories['chicane']
    ax4.plot(np.degrees(example_steering), example_speed, 'o-', markersize=4, linewidth=1, alpha=0.8)
    ax4.set_xlabel('Steering Angle (degrees)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Example: Chicane Maneuver\n(Speed vs Steering)')
    ax4.grid(True, alpha=0.3)

    # Add arrows to show trajectory direction
    for i in range(0, len(example_steering)-10, 20):
        ax4.annotate('',
                    xy=(np.degrees(example_steering[i+10]), example_speed[i+10]),
                    xytext=(np.degrees(example_steering[i]), example_speed[i]),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

    plt.tight_layout()
    plt.savefig('f1tenth_action_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved: f1tenth_action_trajectories.png")
    return fig

def plot_data_comparison():
    """Compare F1TENTH vs CarRacing data formats."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # F1TENTH observations (LiDAR)
    angles, scenarios = create_sample_lidar_data()
    lidar_ranges = scenarios['straight']

    ax1.plot(np.degrees(angles), lidar_ranges, 'b-', linewidth=2)
    ax1.fill_between(np.degrees(angles), lidar_ranges, alpha=0.3, color='blue')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Range (meters)')
    ax1.set_title('F1TENTH Observation\n(LiDAR: 1080 range values)')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'Data shape: ({len(lidar_ranges)},)\nRange: [0.5, 30.0] meters\nNormalized: [0, 1]',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # CarRacing observations (would be camera)
    camera_data = np.random.rand(96, 96, 4)  # Simulated camera frames
    ax2.imshow(camera_data[:,:,0], cmap='gray')
    ax2.set_title('CarRacing Observation\n(Camera: 96x96x4 images)')
    ax2.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels')
    ax2.text(5, 15, f'Data shape: (96, 96, 4)\nRange: [0, 255] RGB\nNormalized: [0, 1]',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # F1TENTH actions
    time_steps, trajectories = create_sample_actions()
    f1_steering, f1_speed = trajectories['straight']

    ax3.plot(time_steps, np.degrees(f1_steering), 'r-', label='Steering (deg)', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_steps, f1_speed, 'g-', label='Speed (m/s)', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Steering Angle (degrees)', color='red')
    ax3_twin.set_ylabel('Speed (m/s)', color='green')
    ax3.set_title('F1TENTH Actions\n[steering_angle, speed]')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.05, 0.95, f'Action dim: 2\nSteering: ±24° (±0.42 rad)\nSpeed: 0-8 m/s',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # CarRacing actions
    car_steer = 0.3 * np.sin(time_steps * 0.8)
    car_gas = 0.5 + 0.3 * np.sin(time_steps * 0.5)
    car_brake = np.maximum(0, -car_gas + 0.5)

    ax4.plot(time_steps, car_steer, 'r-', label='Steer', linewidth=2)
    ax4.plot(time_steps, car_gas, 'g-', label='Gas', linewidth=2)
    ax4.plot(time_steps, car_brake, 'b-', label='Brake', linewidth=2)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Action Value')
    ax4.set_title('CarRacing Actions\n[steer, gas, brake]')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(0.05, 0.95, f'Action dim: 3\nSteer: [-1, +1]\nGas: [0, 1]\nBrake: [0, 1]',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('f1tenth_vs_carracing.png', dpi=150, bbox_inches='tight')
    print("Saved: f1tenth_vs_carracing.png")
    return fig

def create_track_overview():
    """Create an overview of F1 track characteristics."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # F1 track data (simplified characteristics)
    tracks = {
        'Spielberg': {'corners': 10, 'straights': 3, 'elevation': 'High', 'difficulty': 'Medium'},
        'Monaco': {'corners': 19, 'straights': 1, 'elevation': 'Medium', 'difficulty': 'Very Hard'},
        'Monza': {'corners': 11, 'straights': 3, 'elevation': 'Low', 'difficulty': 'Easy'},
        'Silverstone': {'corners': 18, 'straights': 2, 'elevation': 'Medium', 'difficulty': 'Hard'},
        'Spa': {'corners': 20, 'straights': 3, 'elevation': 'High', 'difficulty': 'Hard'},
        'Suzuka': {'corners': 18, 'straights': 2, 'elevation': 'Medium', 'difficulty': 'Hard'}
    }

    for idx, (track_name, data) in enumerate(tracks.items()):
        ax = axes[idx]

        # Create simplified track layout
        theta = np.linspace(0, 2*np.pi, 200)
        # Base oval with track-specific modifications
        r = 1.0 + 0.2 * np.sin(data['corners'] * theta / 2)

        # Add complexity based on corners
        if data['corners'] > 15:  # Technical tracks
            r += 0.1 * np.sin(10 * theta)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Plot track
        ax.fill(x, y, alpha=0.3, color='gray', label='Track')
        ax.plot(x, y, 'k-', linewidth=4, label='Circuit')

        # Add racing line
        racing_r = r * 0.9
        racing_x = racing_r * np.cos(theta)
        racing_y = racing_r * np.sin(theta)
        ax.plot(racing_x, racing_y, 'r--', linewidth=2, label='Racing Line')

        # Start/finish line
        ax.plot([x[0]-0.1, x[0]+0.1], [y[0]-0.1, y[0]+0.1], 'g-', linewidth=6, label='Start/Finish')

        ax.set_aspect('equal')
        ax.set_title(f'{track_name}\nCorners: {data["corners"]}, Difficulty: {data["difficulty"]}')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('f1tenth_tracks_overview.png', dpi=150, bbox_inches='tight')
    print("Saved: f1tenth_tracks_overview.png")
    return fig

def main():
    """Generate all F1TENTH visualizations."""

    print("Generating F1TENTH Environment Visualizations...")
    print("=" * 50)

    # Generate all visualization plots
    print("1. Creating LiDAR scenario visualizations...")
    plot_lidar_scenarios()

    print("2. Creating action trajectory visualizations...")
    plot_action_trajectories()

    print("3. Creating data format comparison...")
    plot_data_comparison()

    print("4. Creating track overview...")
    create_track_overview()

    print("\n✓ All visualizations completed!")
    print("\nGenerated files:")
    print("  - f1tenth_lidar_scenarios.png")
    print("  - f1tenth_action_trajectories.png")
    print("  - f1tenth_vs_carracing.png")
    print("  - f1tenth_tracks_overview.png")

    print("\nThese visualizations show:")
    print("  • How F1TENTH LiDAR observations work")
    print("  • F1TENTH action space and trajectories")
    print("  • Differences from CarRacing environment")
    print("  • Available F1 track layouts")

if __name__ == "__main__":
    main()