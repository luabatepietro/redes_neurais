window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Só processa o que o pymdownx.arithmatex marcou como fórmula
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  svg: {
    // SVG desenha as fórmulas como caminhos: não depende de fontes web,
    // então não some quando o navegador bloqueia/falha ao baixar as fontes.
    fontCache: "global"
  }
};

// Material com navigation.instant: refaz a tipografia a cada troca de página
document$.subscribe(() => {
  if (!window.MathJax?.typesetPromise) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
