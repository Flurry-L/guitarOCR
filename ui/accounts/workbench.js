import { setAuth, element } from "./http.js";
import { authenticate } from "/static/api.js";
const session = await authenticate().catch(() => null);
if (session?.user) {
  setAuth(session);
  const account = element("a", "账号与服务设置", "button");
  account.href = "/#account";
  document
    .querySelector('[data-screen="settings"] .surface-card')
    .append(account);
}
