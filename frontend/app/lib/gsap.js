'use client';

import { gsap } from 'gsap';
import { Draggable } from 'gsap/Draggable';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SplitText } from 'gsap/SplitText';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP, Draggable, ScrollTrigger, SplitText);

// Animations run inside this query, so people who ask for less motion get static pages.
export const MOTION_OK = '(prefers-reduced-motion: no-preference)';

/**
 * Makes a scrubber. Dragging the trigger reports a 0 to 1 position along the track.
 * A detached proxy takes the drag, so React stays in charge of where the handle is drawn.
 */
export function createScrubber(trigger, track, { onScrub, onPress, onRelease }) {
  const report = (draggable) => {
    const rect = track.getBoundingClientRect();
    const pointerX = draggable.pointerX - window.pageXOffset;
    onScrub(gsap.utils.clamp(0, 1, (pointerX - rect.left) / rect.width));
  };

  return Draggable.create(document.createElement('div'), {
    trigger,
    type: 'x',
    cursor: 'grab',
    activeCursor: 'grabbing',
    onPress() {
      onPress?.();
      report(this);
    },
    onDrag() {
      report(this);
    },
    onRelease() {
      onRelease?.();
    },
  })[0];
}

export { gsap, Draggable, ScrollTrigger, SplitText, useGSAP };
