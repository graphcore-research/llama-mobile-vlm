window.highlightCodeBlocks = () => {
  const tokenPattern = /(?<comment>\/\/[^\n]*|#[^\n]*)|(?<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?<number>\b\d+(?:\.\d+)?\b)|(?<keyword>\b(?:and|as|assert|async|await|break|case|catch|class|continue|def|del|do|else|elif|except|false|False|finally|for|from|function|if|in|is|lambda|let|match|new|not|null|None|or|pass|raise|return|switch|tbl|this|throw|true|True|try|ushr|eor|var|while|with|yield)\b)|(?<literal>\b(?:false|False|nil|None|null|true|True|undefined)\b)|(?<type>\b(?:bool|char|double|float|int|long|short|signed|str|string|uint\d*_t|u?int\d*_t|void|unsigned|size_t|auto|const|static|struct|enum|class|typename|vector|std)\b)|(?<function>\b[A-Za-z_$][\w$]*(?=\s*\())|(?<operator>==|!=|<=|>=|->|=>|\+\+|--|&&|\|\||\+=|-=|\*=|\/=|[=+*/%<>!&|^~?:-])/g;
  const escapeHtml = text => text.replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);

  document.querySelectorAll('pre code:not([data-highlighted])').forEach(code => {
    const source = code.textContent;
    let cursor = 0;
    const highlighted = [];
    for (const match of source.matchAll(tokenPattern)) {
      highlighted.push(escapeHtml(source.slice(cursor, match.index)));
      const kind = Object.keys(match.groups).find(key => match.groups[key] !== undefined);
      highlighted.push(`<span class="token-${kind}">${escapeHtml(match[0])}</span>`);
      cursor = match.index + match[0].length;
    }
    highlighted.push(escapeHtml(source.slice(cursor)));
    code.innerHTML = highlighted.join('');
    code.dataset.highlighted = 'true';
  });
};

new MutationObserver(() => window.highlightCodeBlocks()).observe(
  document.querySelector('.reveal .slides'),
  { childList: true, subtree: true }
);
