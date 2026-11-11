
export function initSidebar() {
  const container = document.getElementById("sidebar-container");

  if (!container) return;

  const normalizePath = (path) => {
    const cleanPath = path
      .split("?")[0]
      .split("#")[0]
      .replace(/\/+$/, "")
      .toLowerCase();

    return cleanPath || "/";
  };

  const currentPath = normalizePath(
    window.location.pathname
  );

  const links = container.querySelectorAll(
    ".nav-menu a.nav-item"
  );

  links.forEach((link) => {
    const href = link.getAttribute("href");

    if (!href) return;

    const linkPath = normalizePath(
      new URL(href, window.location.origin).pathname
    );

    const isActive = currentPath === linkPath;

    link.classList.toggle("active", isActive);

    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}
