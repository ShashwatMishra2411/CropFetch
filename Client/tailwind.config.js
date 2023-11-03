/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#C6B09D",
        secondary: "#37322C",
        accent: "#847B65",
        highlight: "#FFA300",
      },
      fontFamily: {
        raleway: "Raleway, sans-serif",
        dots: "Raleway Dots, cursive",
      },
    },
  },
  plugins: [],
};
