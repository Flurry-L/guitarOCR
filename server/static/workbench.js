import { setAuth, element } from "./http.js";
const response = await fetch("/api/auth/me");
const session = response.ok ? await response.json() : null;
if (!session?.user) location.replace("/");
else {
  setAuth(session);
  const back = element("a", "返回账号");
  back.href = "/";
  document.querySelector(".app-header").append(back);
  document.querySelector(".project-import").hidden = true;
}
