import React, {useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Display from './Display'

gsap.registerPlugin(ScrollTrigger);

export default function Scroll() {

  return (
    <Display/>
  );
}
