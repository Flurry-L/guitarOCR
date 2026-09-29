import { setAuth, element } from "./http.js";
const response = await fetch("/api/auth/me");
const session = response.ok ? await response.json() : null;
if (!session?.user) location.replace("/");
else {
  setAuth(session);
  const account = element("a", "账号与服务设置", "button");
  account.href = "/#account";
  document
    .querySelector('[data-screen="settings"] .surface-card')
    .append(account);
  document.querySelector(".project-import").hidden = true;
}
