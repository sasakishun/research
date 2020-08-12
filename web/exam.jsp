<%@ page language="java" contentType="text/html;charset=Windows-31J"%>
<%@ page import="java.util.*" %>
<%@ page import="smp.SampleBean" %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>

<html><body>

<%
List list = new ArrayList();
SampleBean bean = new SampleBean();
bean.setName("田中");
list.add(bean);

bean = new SampleBean();
bean.setName("山本");
list.add(bean);

bean = new SampleBean();
bean.setName("鈴木");
list.add(bean);

request.setAttribute("data",list);
%>

<c:forEach var="obj" items="${data}" varStatus="status">
　　名前：<c:out value="${obj.name}"/><br>
　　index：<c:out value="${status.index}"/><br>
</c:forEach>

</body></html>