/**
 * CartPole visualizer — renders the 4D state on a 2D canvas.
 *
 * State: [cart position, cart velocity, pole angle, pole angular velocity]
 * - Cart position is in [-2.4, 2.4], centered in the canvas
 * - Pole angle is in radians, 0 = upright, |theta| < 12° before termination
 * - Action 0 = push left, action 1 = push right (rendered as a brief arrow)
 */

const TRACK_HALF_WIDTH = 2.4;
const TRACK_Y = 160;
const CART_WIDTH = 40;
const CART_HEIGHT = 20;
const POLE_LENGTH_PX = 90;

export class CartPoleRenderer {
  private ctx: CanvasRenderingContext2D;
  private width: number;
  private height: number;
  private lastActionTime = 0;

  constructor(canvasId: string) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    this.ctx = canvas.getContext('2d')!;
    this.width = canvas.width;
    this.height = canvas.height;
    this.clear();
  }

  clear(): void {
    this.ctx.fillStyle = '#1c2230';
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.drawTrack();
  }

  /**
   * Render the current state.
   * @param state - CartPole 4D state [x, xDot, theta, thetaDot]
   * @param action - Last action taken (0 or 1), optional
   */
  render(state: Float32Array, action?: number): void {
    if (action !== undefined) {
      this.lastActionTime = performance.now();
    }
    const [x = 0, , theta = 0] = state;
    this.clear();
    this.drawCart(x);
    this.drawPole(x, theta);
    if (action !== undefined && performance.now() - this.lastActionTime < 200) {
      this.drawActionArrow(x, action);
    }
  }

  private drawTrack(): void {
    this.ctx.strokeStyle = '#3a4458';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(20, TRACK_Y);
    this.ctx.lineTo(this.width - 20, TRACK_Y);
    this.ctx.stroke();

    // Draw termination zone markers
    this.ctx.strokeStyle = '#ff5d5d55';
    this.ctx.setLineDash([4, 4]);
    this.ctx.beginPath();
    const leftX = this.stateToScreenX(-TRACK_HALF_WIDTH);
    const rightX = this.stateToScreenX(TRACK_HALF_WIDTH);
    this.ctx.moveTo(leftX, 30);
    this.ctx.lineTo(leftX, TRACK_Y + 10);
    this.ctx.moveTo(rightX, 30);
    this.ctx.lineTo(rightX, TRACK_Y + 10);
    this.ctx.stroke();
    this.ctx.setLineDash([]);
  }

  private drawCart(x: number): void {
    const cartX = this.stateToScreenX(x);
    const cartY = TRACK_Y - CART_HEIGHT / 2;
    this.ctx.fillStyle = '#4c8eff';
    this.ctx.fillRect(cartX - CART_WIDTH / 2, cartY, CART_WIDTH, CART_HEIGHT);
    this.ctx.strokeStyle = '#6da3ff';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(cartX - CART_WIDTH / 2, cartY, CART_WIDTH, CART_HEIGHT);
  }

  private drawPole(x: number, theta: number): void {
    const cartX = this.stateToScreenX(x);
    const cartTopY = TRACK_Y - CART_HEIGHT;
    // Pole hangs from the cart top, tilted by theta
    const tipX = cartX + Math.sin(theta) * POLE_LENGTH_PX;
    const tipY = cartTopY - Math.cos(theta) * POLE_LENGTH_PX;

    // Color shifts blue→red as |theta| grows
    const angleNorm = Math.min(1, Math.abs(theta) / (12 * Math.PI / 180));
    const r = Math.round(76 + (255 - 76) * angleNorm);
    const g = Math.round(142 + (93 - 142) * angleNorm);
    const b = Math.round(255 + (93 - 255) * angleNorm);
    this.ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
    this.ctx.lineWidth = 4;
    this.ctx.lineCap = 'round';
    this.ctx.beginPath();
    this.ctx.moveTo(cartX, cartTopY);
    this.ctx.lineTo(tipX, tipY);
    this.ctx.stroke();
    this.ctx.lineCap = 'butt';
  }

  private drawActionArrow(x: number, action: number): void {
    const cartX = this.stateToScreenX(x);
    const arrowX = action === 1 ? cartX + 30 : cartX - 30;
    const direction = action === 1 ? 1 : -1;
    this.ctx.strokeStyle = '#3ddc97';
    this.ctx.fillStyle = '#3ddc97';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(cartX, TRACK_Y + 6);
    this.ctx.lineTo(arrowX, TRACK_Y + 6);
    this.ctx.stroke();
    // Arrow head
    this.ctx.beginPath();
    this.ctx.moveTo(arrowX, TRACK_Y + 6);
    this.ctx.lineTo(arrowX - 6 * direction, TRACK_Y + 3);
    this.ctx.lineTo(arrowX - 6 * direction, TRACK_Y + 9);
    this.ctx.closePath();
    this.ctx.fill();
  }

  private stateToScreenX(x: number): number {
    const usable = this.width - 40;
    const ratio = (x + TRACK_HALF_WIDTH) / (2 * TRACK_HALF_WIDTH);
    return 20 + ratio * usable;
  }
}
