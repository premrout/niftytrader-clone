ce_ltp = "" if is_total else (f"{r.get('CE_LTP'):.2f}" if r.get('CE_LTP') else "")
pe_ltp = "" if is_total else (f"{r.get('PE_LTP'):.2f}" if r.get('PE_LTP') else "")

rows_html.append(f"""
<tr style="{bg}">
  <td style="text-align:center;width:90px">{s}</td>
  <td style="text-align:right;width:120px">{ce_disp} {ce_bar}</td>
  <td style="text-align:right;width:90px">{ce_ltp}</td>
  <td style="text-align:center;width:60px"></td>
  <td style="text-align:right;width:90px">{pe_ltp}</td>
  <td style="text-align:left;width:120px">{pe_disp} {pe_bar}</td>
  <td style="text-align:right;width:120px">{diff_html}</td>
</tr>
""")
