for ($i=63; $i -ge 26; $i=$i-1 ) {
$p = $i + 1
$pre = "{0}_" -f $i
$ren = "{0}_" -f $p
get-childitem * | foreach { rename-item $_ $_.Name.Replace($pre, $ren) }
}